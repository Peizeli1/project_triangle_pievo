#!/usr/bin/env python3
"""Local OpenAI-compatible bridge for Gemini-2.5-Flash-Lite with thinking disabled.

The New API gateway exposes Gemini's native endpoint at ``/v1beta``.  Its
OpenAI-compatible endpoint does not reliably honor a generic
``reasoning_effort`` value for Gemini.  This bridge always calls the native
endpoint with ``thinkingBudget=0`` and fails closed if the returned usage
metadata reports positive thought tokens.  Flash-Lite may omit
``thoughtsTokenCount`` when no thinking channel is present; that response shape
is recorded separately as a budget-zero/field-absent attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from gemini_model_config import NATIVE_MODEL
except ModuleNotFoundError:
    from experiments.gemini_model_config import NATIVE_MODEL
from urllib.parse import urlsplit


def read_key(path: Path | None) -> str:
    """Read an explicit key file when present, otherwise use the local helper.

    The helper resolves the workstation Keychain entry without copying the
    credential into a run directory or manifest.
    """
    try:
        from runtime_credentials import read_api_key
    except ModuleNotFoundError:
        from experiments.runtime_credentials import read_api_key
    return read_api_key(path)


def endpoint_from_base_url(base_url: str, model: str) -> str:
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid upstream base URL: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}/v1beta/models/{model}:generateContent"


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                values.append(item["text"])
        return "\n".join(values)
    return "" if content is None else str(content)


def schema_for_gemini(value: Any) -> Any:
    if isinstance(value, list):
        return [schema_for_gemini(item) for item in value]
    if not isinstance(value, dict):
        return value
    converted: dict[str, Any] = {}
    for key, item in value.items():
        if key == "type" and isinstance(item, str):
            converted[key] = item.upper()
        elif key in {"properties", "items", "anyOf", "oneOf", "allOf"}:
            converted[key] = schema_for_gemini(item)
        elif key not in {"additionalProperties", "$schema", "$defs"}:
            converted[key] = item
    return converted


def gemini_tools(openai_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(openai_tools, list):
        return None
    declarations: list[dict[str, Any]] = []
    for item in openai_tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        declaration = {"name": function["name"]}
        if isinstance(function.get("description"), str):
            declaration["description"] = function["description"]
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else None


def gemini_contents(messages: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = text_from_content(message.get("content"))
        if role == "system":
            if content:
                system_parts.append({"text": content})
            continue
        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": str(arguments)}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {"value": parsed_arguments}
                call_id = str(call.get("id", uuid.uuid4().hex))
                tool_names[call_id] = function["name"]
                parts.append({"functionCall": {"name": function["name"], "args": parsed_arguments}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            name = str(message.get("name") or tool_names.get(call_id) or "tool")
            try:
                response_value = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                response_value = {"result": content}
            if not isinstance(response_value, dict):
                response_value = {"result": response_value}
            contents.append(
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": name, "response": response_value}}],
                }
            )
            continue
        contents.append({"role": "user", "parts": [{"text": content}]})
    if not contents:
        raise ValueError("no usable message content")
    system = {"parts": system_parts} if system_parts else None
    return contents, system


def normalize_function_arguments(arguments: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Undo a deterministic JSON-string double encoding in tool arguments.

    Some Flash-Lite function calls place a scalar string such as a SMILES
    value inside an additional JSON string layer (``"\\\"CCO\\\""``).  The
    OpenAI tool schema expects the scalar itself.  This transport repair is
    outcome-blind and only applies when decoding the value produces exactly
    one string scalar; objects, arrays, numbers, and ordinary strings are
    left untouched.
    """
    if not isinstance(arguments, dict):
        return arguments, []
    normalized: dict[str, Any] = {}
    adaptations: list[dict[str, Any]] = []
    for key, value in arguments.items():
        replacement = value
        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
                try:
                    decoded = json.loads(stripped)
                except (TypeError, json.JSONDecodeError):
                    decoded = None
                if isinstance(decoded, str):
                    replacement = decoded
                    adaptations.append(
                        {
                            "field": str(key),
                            "kind": "json_string_double_encoding",
                            "original": value,
                            "normalized": decoded,
                        }
                    )
        normalized[key] = replacement
    return normalized, adaptations


class GeminiNoThinkingClient:
    def __init__(self, *, upstream_base_url: str, api_key: str, model: str, timeout: float, max_transport_retries: int):
        self.endpoint = endpoint_from_base_url(upstream_base_url, model)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_transport_retries = max(0, max_transport_retries)

    def complete(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        contents, system_instruction = gemini_contents(request.get("messages"))
        generation: dict[str, Any] = {
            "thinkingConfig": {"thinkingBudget": 0, "includeThoughts": False},
        }
        if isinstance(request.get("temperature"), (int, float)):
            generation["temperature"] = request["temperature"]
        if isinstance(request.get("top_p"), (int, float)):
            generation["topP"] = request["top_p"]
        if isinstance(request.get("seed"), int) and not isinstance(request.get("seed"), bool):
            generation["seed"] = request["seed"]
        max_tokens = request.get("max_completion_tokens", request.get("max_tokens"))
        if isinstance(max_tokens, int) and max_tokens > 0:
            generation["maxOutputTokens"] = max_tokens
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction
        tools = gemini_tools(request.get("tools"))
        if tools:
            body["tools"] = tools
            tool_choice = request.get("tool_choice")
            if tool_choice != "none":
                function_config: dict[str, Any] = {"mode": "ANY"}
                if isinstance(tool_choice, dict):
                    function = tool_choice.get("function", {})
                    if isinstance(function, dict) and isinstance(function.get("name"), str):
                        function_config["allowedFunctionNames"] = [function["name"]]
                body["toolConfig"] = {"functionCallingConfig": function_config}
        response_format = request.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") in {"json_object", "json_schema"}:
            generation["responseMimeType"] = "application/json"
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        native_request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        retryable_statuses = {429, 500, 502, 503, 504}
        transport_attempts = 0
        rejected_thought_token_counts: list[int | None] = []
        while True:
            transport_attempts += 1
            try:
                with urllib.request.urlopen(native_request, timeout=self.timeout) as handle:
                    payload = json.loads(handle.read().decode("utf-8"))
                usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
                thoughts = usage.get("thoughtsTokenCount")
                # Flash-Lite omits thoughtsTokenCount when the thinking
                # channel is disabled.  The request carries the explicit
                # budget-zero/includeThoughts=false contract.
                if thoughts in {0, None}:
                    break
                rejected_thought_token_counts.append(
                    int(thoughts) if isinstance(thoughts, int) and not isinstance(thoughts, bool) else None
                )
                if transport_attempts > self.max_transport_retries:
                    raise RuntimeError(
                        "Gemini no-thinking violation after discarded transport responses: "
                        f"thoughtsTokenCount={thoughts!r}"
                    )
                time.sleep(min(2.0 ** (transport_attempts - 1), 4.0))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in retryable_statuses or transport_attempts > self.max_transport_retries:
                    raise RuntimeError(f"Gemini native request failed ({exc.code}): {detail}") from exc
                time.sleep(min(2.0 ** (transport_attempts - 1), 4.0))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if transport_attempts > self.max_transport_retries:
                    raise RuntimeError(f"Gemini native transport failed: {exc}") from exc
                time.sleep(min(2.0 ** (transport_attempts - 1), 4.0))
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        return payload, {
            "request_body": body,
            "usage": usage,
            "transport_attempts": transport_attempts,
            "discarded_thought_token_counts": rejected_thought_token_counts,
        }


class BridgeState:
    def __init__(self, client: GeminiNoThinkingClient, audit_log: Path):
        self.client = client
        self.audit_log = audit_log
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def completion_response(request: dict[str, Any], native: dict[str, Any], usage: dict[str, Any], model: str) -> dict[str, Any]:
    candidates = native.get("candidates", []) if isinstance(native, dict) else []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    content_data = candidate.get("content", {}) if isinstance(candidate, dict) else {}
    parts = content_data.get("parts", []) if isinstance(content_data, dict) else []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    argument_adaptations: list[dict[str, Any]] = []
    for index, part in enumerate(parts if isinstance(parts, list) else []):
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        function = part.get("functionCall")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            normalized_args, adaptations = normalize_function_arguments(function.get("args", {}))
            argument_adaptations.extend(
                [{"tool_name": function["name"], **item} for item in adaptations]
            )
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(normalized_args, ensure_ascii=False),
                    },
                }
            )
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    completion_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    reasoning_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason = "tool_calls" if tool_calls else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "_bridge_argument_adaptations": argument_adaptations,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens + reasoning_tokens,
            "total_tokens": int(usage.get("totalTokenCount", prompt_tokens + completion_tokens + reasoning_tokens) or 0),
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def make_handler(state: BridgeState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GeminiNoThinkingBridge/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self.send_json(HTTPStatus.OK, {"object": "list", "data": [{"id": state.client.model, "object": "model"}]})
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found", "type": "invalid_request_error"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if request.get("stream") is True:
                    raise ValueError("streaming is disabled for the Gemini native bridge")
                native, details = state.client.complete(request)
                response = completion_response(request, native, details["usage"], state.client.model)
                state.log(
                    {
                        "schema_version": "gemini_no_thinking_bridge_call.v1",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "model_requested": request.get("model"),
                        "native_model": state.client.model,
                        "thinking_budget": 0,
                        "temperature_requested": request.get("temperature"),
                        "max_tokens_requested": request.get("max_completion_tokens", request.get("max_tokens")),
                        "top_p_requested": request.get("top_p"),
                        "enable_thinking": False,
                        "seed_requested": request.get("seed"),
                        "seed_forwarded": isinstance(request.get("seed"), int)
                        and not isinstance(request.get("seed"), bool),
                        "transport_attempts": details.get("transport_attempts"),
                        "discarded_thought_token_counts": details.get(
                            "discarded_thought_token_counts", []
                        ),
                        "thoughts_token_count": details["usage"].get("thoughtsTokenCount"),
                        "no_thinking_verified": details["usage"].get("thoughtsTokenCount") in {0, None},
                        "no_thinking_attestation": (
                            "usage_zero"
                            if details["usage"].get("thoughtsTokenCount") == 0
                            else "budget_zero_usage_field_absent"
                        ),
                        "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
                        "usage": response["usage"],
                        "tool_call_count": len(response["choices"][0]["message"].get("tool_calls", [])),
                        "tool_argument_adaptations": response.get("_bridge_argument_adaptations", []),
                    }
                )
                self.send_json(HTTPStatus.OK, response)
            except Exception as exc:
                state.log(
                    {
                        "schema_version": "gemini_no_thinking_bridge_call.v1",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "native_model": state.client.model,
                        "thinking_budget": 0,
                        "model_requested": request.get("model"),
                        "temperature_requested": request.get("temperature"),
                        "max_tokens_requested": request.get("max_completion_tokens", request.get("max_tokens")),
                        "top_p_requested": request.get("top_p"),
                        "enable_thinking": False,
                        "no_thinking_verified": False,
                        "error": str(exc),
                    }
                )
                self.send_json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(exc), "type": "upstream_error"}})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-base-url", default="https://zjuapi.com/v1")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--model", default=NATIVE_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-transport-retries", type=int, default=2)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("bridge must bind only to a loopback address")
    client = GeminiNoThinkingClient(
        upstream_base_url=args.upstream_base_url,
        api_key=read_key(args.api_key_file),
        model=args.model,
        timeout=args.timeout,
        max_transport_retries=args.max_transport_retries,
    )
    state = BridgeState(client, args.audit_log)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    bound_port = int(server.server_address[1])
    print(json.dumps({"event": "ready", "base_url": f"http://{args.host}:{bound_port}/v1", "model": args.model, "thinking_budget": 0}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
