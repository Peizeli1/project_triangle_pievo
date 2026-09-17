#!/usr/bin/env python3
"""Run one auditable non-PiEvo baseline cell for the Qwen Table 1 track.

The three reference adaptations under ``Downloads/baselines`` use an old HTTP
surrogate interface and hard-code legacy model credentials.  This runner keeps
their search-loop semantics and prompt profiles, but binds every method to the
released local PiEvo/PiFlow evaluator and the current Qwen endpoint.  It is a
separate track from the matched PiEvo/RADP formal runner, so existing formal
artifacts are never overwritten or spliced into a baseline run.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MATCHED_REPO = ROOT / "official_sources/PiEvo_matched_qwen235b"
BASELINE_ROOT = Path("/Users/michael/Downloads/baselines")
CONTROLLED_PROTOCOL_PATH = ROOT / "experiments/configs/controlled_qwen235b_protocol_v1.json"
TASKS = ("MBO", "NHO", "SPO", "TMC")
METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def load_protocol(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"controlled protocol is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"controlled protocol must be a JSON object: {path}")
    required = ("temperature", "max_tokens", "top_p", "top_k", "presence_penalty", "repetition_penalty", "enable_thinking", "context_messages")
    missing = [key for key in required if key not in value]
    if missing:
        raise SystemExit(f"controlled protocol missing fields: {', '.join(missing)}")
    return value

TMC_NEUTRAL = (
    "RUCBEY-subgraph-1",
    "WECJIA-subgraph-3",
    "KEYRUB-subgraph-1",
    "NURKEQ-subgraph-2",
    "MEBXUN-subgraph-1",
    "BIFMOV-subgraph-1",
    "CUJYEL-subgraph-2",
    "EZEXEM-subgraph-1",
    "FOMVUB-subgraph-2",
    "EFIHEJ-subgraph-3",
    "LETTEL-subgraph-1",
    "KAKKIR-subgraph-3",
    "BICRIQ-subgraph-3",
    "UPEGAZ-subgraph-2",
    "CEVJAP-subgraph-2",
    "BABTUT-subgraph-3",
    "ZEJJEF-subgraph-3",
    "KULGAZ-subgraph-2",
    "CIGDAA-subgraph-1",
    "HOVMIP-subgraph-3",
    "ULUSIE-subgraph-1",
    "IBEKUV-subgraph-1",
    "REQSUD-subgraph-2",
    "BOSJIF-subgraph-1",
    "GUVMEP-subgraph-0",
)
TMC_ANION = (
    "MAZJIJ-subgraph-0",
    "OBONEA-subgraph-1",
    "CORTOU-subgraph-2",
    "LEVGUO-subgraph-2",
    "REBWEB-subgraph-2",
    "DOGPAS-subgraph-1",
    "IJIMIX-subgraph-1",
    "PEJGAN-subgraph-1",
    "BIFZEX-subgraph-0",
    "IRIXUC-subgraph-3",
    "SAYGOO-subgraph-0",
    "UROGIS-subgraph-1",
    "MAQKEX-subgraph-1",
    "LUQWUQ-subgraph-1",
    "QAYDID-subgraph-2",
    "MOYDOV-subgraph-3",
    "NIZQUK-subgraph-1",
    "SAYHIJ-subgraph-1",
    "CIQGOY-subgraph-0",
    "VUFZUT-subgraph-1",
    "ZOQFIU-subgraph-0",
    "GUQBUQ-subgraph-0",
    "LEZYUM-subgraph-2",
    "RAJXUX-subgraph-2",
    "QEWZOH-subgraph-3",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(method: str) -> dict[str, str | None]:
    paths: list[Path] = [Path(__file__)]
    if method == "piflow":
        paths.extend((ROOT / "official_sources/PiFlow").glob("**/*.py"))
    else:
        name = {
            "ai_scientist_v1": "ai_scientist_v1_adapt",
            "ai_scientist_v2": "ai_scientist_v2_adapt",
            "ai_researcher": "ai_researcher_adapt",
        }.get(method)
        if name:
            paths.extend((BASELINE_ROOT / name).glob("**/*.py"))
    return {str(path): sha256(path) for path in sorted(set(paths)) if path.is_file()}


def model_track_for(model: str) -> str:
    """Return a stable table-1 track identifier for the selected model."""
    if model == "qwen3-235b-a22b-instruct-2507":
        return "qwen3-235b-table1"
    return f"{model}-table1"


def is_primary_result(result: dict[str, Any]) -> bool:
    """Normalize the MBO adapter's legacy audit schema to a primary row."""
    return bool(
        result.get("success") is True
        and isinstance(result.get("output"), (int, float))
        and not isinstance(result.get("output"), bool)
        and math.isfinite(float(result["output"]))
        and result.get("record_type") not in {"duplicate_suppressed", "verification"}
        and result.get("duplicate_suppressed") is not True
    )


def read_api_key(path: Path | None) -> str:
    from runtime_credentials import read_api_key as read_runtime_api_key

    return read_runtime_api_key(path)


def task_prompt(task: str) -> str:
    if task == "MBO":
        return (
            "MBO: propose one neutral, connected, RDKit-parseable SMILES string. "
            "Use only C,N,O,S,F,Cl,Br; no formal charges, isotopes, stereochemistry, "
            "or multiple fragments. Maximize the official pChEMBL surrogate."
        )
    if task == "NHO":
        return (
            "NHO: propose exactly one string with fields "
            "fiber_radius, helix_radius, n_turns, pitch, wavelength, x_y_z, direction. "
            "Ranges are [20,60], [20,90], [3,10], [60,200], [400,800], x_y_z in "
            "{0,1,2}, direction in {0,1}."
        )
    if task == "SPO":
        return (
            "SPO: propose exactly one formula using only one of these element sets: "
            "La-Sr-Cu-O, La-Ba-Cu-O, Y-Ba-Cu-O, Bi-Sr-Ca-Cu-O, "
            "Tl-Ba-Ca-Cu-O, Hg-Ba-Ca-Cu-O. Use element symbols and integer ratios only."
        )
    return (
        "TMC: propose exactly one Pd_L1_L2_L3_L4 string. Each ligand must come from "
        "the released pool and the four charges must sum to -2. Use two -1 ligands "
        "and two neutral ligands."
    )


def tmc_pool_prompt() -> str:
    return "Neutral IDs: " + ", ".join(TMC_NEUTRAL) + ". Anionic IDs: " + ", ".join(TMC_ANION) + "."


def fallback_candidates(task: str) -> Iterable[str]:
    if task == "MBO":
        values = [
            "c1ccccc1",
            "Cc1ccccc1",
            "CCc1ccccc1",
            "Oc1ccccc1",
            "COc1ccccc1",
            "Nc1ccccc1",
            "Fc1ccccc1",
            "Clc1ccccc1",
            "Brc1ccccc1",
            "N#Cc1ccccc1",
            "NC(=O)c1ccccc1",
            "O=C(O)c1ccccc1",
            "c1ccc2ccccc2c1",
            "Cc1ccc2ccccc2c1",
            "Oc1ccc2ccccc2c1",
            "COc1ccc2ccccc2c1",
            "c1ccc2c(c1)ccc1ccccc12",
            "c1ccc2c(c1)[nH]c1ccccc12",
            "c1ccc2c(c1)oc1ccccc12",
            "c1ccc2c(c1)sc1ccccc12",
            "C1CCCCCCC1",
            "C1CCCCCCCCCCC1",
            "C1CCC2CCCCC2C1",
            "C1C2CC3CC1CC(C2)C3",
            "O=C1CCCCCCCCCCC1",
            "COC(=O)c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CCOc1ccc(C)cc1",
            "CCN(CC)Cc1ccccc1",
            "O=c1[nH]c(=O)[nH]c(=O)[nH]1",
        ]
        yield from values
        return
    if task == "NHO":
        for index in range(80):
            fiber = 20.0 + float((index * 7) % 41)
            helix = 20.0 + float((index * 13) % 71)
            turns = 3.0 + float((index * 3) % 71) / 10.0
            pitch = 60.0 + float((index * 17) % 141)
            wavelength = 400.0 + float((index * 23) % 401)
            yield (
                f"fiber_radius={fiber:.1f}, helix_radius={helix:.1f}, "
                f"n_turns={turns:.1f}, pitch={pitch:.1f}, "
                f"wavelength={wavelength:.1f}, x_y_z={index % 3}, direction={(index // 3) % 2}"
            )
        return
    if task == "SPO":
        systems = (
            ("La", "Sr", "Cu"),
            ("La", "Ba", "Cu"),
            ("Y", "Ba", "Cu"),
            ("Bi", "Sr", "Ca", "Cu"),
            ("Tl", "Ba", "Ca", "Cu"),
            ("Hg", "Ba", "Ca", "Cu"),
        )
        for index in range(80):
            # Keep the deterministic repair pool unique for the full budget.
            # The old modulo periods repeated after 12 formulas, so a model
            # that kept proposing duplicates could never reach 24 primary
            # evaluations.
            block, slot = divmod(index, len(systems))
            system = systems[slot]
            numbers = [1 + ((block + slot + j * 2) % 9) for j in range(len(system))]
            oxygen = 5 + block
            yield "".join(f"{element}{number}" for element, number in zip(system, numbers)) + f"O{oxygen}"
        return
    seen: set[str] = set()
    for neutral_a, neutral_b in itertools.combinations(TMC_NEUTRAL, 2):
        for anion_a, anion_b in itertools.combinations(TMC_ANION, 2):
            candidate = f"Pd_{neutral_a}_{neutral_b}_{anion_a}_{anion_b}"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def normalize_nho(value: Any) -> str | None:
    if isinstance(value, dict):
        aliases = {"x_y": "x_y_z", "forward_backward": "direction"}
        values = {aliases.get(str(key), str(key)): item for key, item in value.items()}
        fields = ("fiber_radius", "helix_radius", "n_turns", "pitch", "wavelength", "x_y_z", "direction")
        if all(field in values for field in fields):
            return ", ".join(f"{field}={values[field]}" for field in fields)
    text = str(value).strip().strip("`\"'")
    match = re.search(
        r"fiber[_ -]?radius\s*=\s*[-+]?\d+(?:\.\d+)?[\s,;]+"
        r"helix[_ -]?radius\s*=\s*[-+]?\d+(?:\.\d+)?[\s,;]+"
        r"n[_ -]?turns\s*=\s*[-+]?\d+(?:\.\d+)?[\s,;]+"
        r"pitch\s*=\s*[-+]?\d+(?:\.\d+)?[\s,;]+"
        r"wavelength\s*=\s*[-+]?\d+(?:\.\d+)?[\s,;]+"
        r"(?:x[_ -]?y[_ -]?z|x[_ -]?y)\s*=\s*\d+[\s,;]+"
        r"(?:direction|forward[_ -]?backward)\s*=\s*\d+",
        text,
        re.IGNORECASE,
    )
    return match.group(0).replace(";", ",") if match else None


def candidate_values(value: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key in ("candidate", "experimental_candidate", "hypothesis", "action", "input", "value"):
            if key in value:
                values.extend(candidate_values(value[key]))
        for item in value.values():
            if isinstance(item, (dict, list)):
                values.extend(candidate_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(candidate_values(item))
    elif value is not None:
        values.append(value)
    return values


def extract_candidate(text: str, task: str) -> str | None:
    raw = str(text or "").strip()
    candidates: list[Any] = []
    for match in re.finditer(r"<candidate>(.*?)</candidate>", raw, re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        try:
            candidates.extend(candidate_values(json.loads(block)))
        except json.JSONDecodeError:
            candidates.append(block)
    for block in (raw,):
        try:
            candidates.extend(candidate_values(json.loads(block)))
        except json.JSONDecodeError:
            pass
    candidates.extend(re.findall(r"(?:candidate|hypothesis|action(?:_input)?)\s*[:=]\s*([^\n]+)", raw, re.IGNORECASE))
    candidates.extend(line.strip() for line in raw.splitlines())

    for value in candidates:
        if isinstance(value, dict):
            value = normalize_nho(value) if task == "NHO" else None
        if value is None:
            continue
        text_value = str(value).strip().strip("`\"'")
        if task == "NHO":
            normalized = normalize_nho(text_value)
            if normalized:
                return normalized
        elif task == "TMC":
            match = re.search(r"Pd_[A-Za-z0-9-]+(?:_[A-Za-z0-9-]+){3}", text_value)
            if match:
                return match.group(0)
        elif task == "SPO":
            match = re.search(r"(?:La|Sr|Y|Ba|Bi|Ca|Cu|Tl|Hg|O)\d*", text_value)
            if match and re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", text_value):
                return text_value
        elif task == "MBO":
            if " " not in text_value and len(text_value) <= 160 and re.fullmatch(r"[A-Za-z0-9@+\-()\[\]=#$\\/.:%]+", text_value):
                return text_value
    return None


class LLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        seed: int,
        output: Path,
        temperature: float,
        max_tokens: int,
        protocol: dict[str, Any],
    ):
        from openai import OpenAI

        # Keep the client timeout above the bridge's bounded upstream timeout;
        # otherwise a slow but valid native response is failed locally first.
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=1, timeout=150.0)
        self.model = model
        self.seed = seed
        self.output = output
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.protocol = protocol
        self.call_index = 0
        self.seed_supported = True

    def effective_generation_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.protocol["top_p"],
            "top_k": self.protocol["top_k"],
            "presence_penalty": self.protocol["presence_penalty"],
            "repetition_penalty": self.protocol["repetition_penalty"],
            "enable_thinking": self.protocol["enable_thinking"],
            "reasoning_effort": self.protocol.get("reasoning_effort"),
        }

    def generate(self, *, method: str, task: str, purpose: str, prompt: str) -> str:
        self.call_index += 1
        started = time.time()
        is_gemini = self.model.lower().startswith("gemini-")
        request_prompt = prompt
        if is_gemini:
            request_prompt += (
                "\nReturn your exact response by calling emit_response once; "
                "put the full response text in its text field."
            )
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise scientific black-box optimization agent. Return a candidate that obeys the requested schema."},
                {"role": "user", "content": request_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.protocol["top_p"],
            "seed": self.seed,
        }
        if not is_gemini and self.model.lower().startswith("qwen"):
            # Match the PiEvo client extras and explicitly disable Qwen thinking
            # for baseline requests.  These fields are part of the effective
            # request, not merely a prompt-level convention.
            request["extra_body"] = {
                "top_k": self.protocol["top_k"],
                "presence_penalty": self.protocol["presence_penalty"],
                "repetition_penalty": self.protocol["repetition_penalty"],
                "chat_template_kwargs": {
                    "enable_thinking": self.protocol["enable_thinking"]
                },
            }
        if is_gemini:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "emit_response",
                        "description": "Return the exact response text requested by the user prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                }
            ]
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": "emit_response"},
            }
        if self.model.lower().startswith("gpt-5"):
            request["reasoning_effort"] = self.protocol.get("reasoning_effort", "medium")
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if "seed" not in str(exc).lower():
                raise
            self.seed_supported = False
            request.pop("seed", None)
            response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        content = str(message.content or "")
        if is_gemini and not content:
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                raw_arguments = getattr(tool_calls[0].function, "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                if isinstance(arguments, dict):
                    content = str(arguments.get("text", ""))
        usage = getattr(response, "usage", None)
        row = {
            "call_index": self.call_index,
            "method": method,
            "task": task,
            "purpose": purpose,
            "request": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.protocol["top_p"],
                "top_k": self.protocol["top_k"] if not is_gemini and self.model.lower().startswith("qwen") else None,
                "presence_penalty": self.protocol["presence_penalty"] if not is_gemini and self.model.lower().startswith("qwen") else None,
                "repetition_penalty": self.protocol["repetition_penalty"] if not is_gemini and self.model.lower().startswith("qwen") else None,
                "enable_thinking": self.protocol["enable_thinking"] if not is_gemini and self.model.lower().startswith("qwen") else False if is_gemini else None,
                "reasoning_effort": self.protocol.get("reasoning_effort") if self.model.lower().startswith("gpt-5") else None,
                "seed": self.seed if self.seed_supported else None,
                "gemini_function_envelope": is_gemini,
            },
            "prompt": request_prompt,
            "response": content,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            "elapsed_seconds": round(time.time() - started, 4),
        }
        with (self.output / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return content


def build_prompt(method: str, task: str, history: list[dict[str, Any]], parent: str | None, analysis: str = "") -> str:
    history_text = "\n".join(
        f"{item['iteration']}. candidate={item['candidate']!r}; outcome={item.get('output')}; status={item.get('status')}"
        for item in history[-10:]
    ) or "No candidates have been evaluated yet."
    schema = task_prompt(task)
    pool = f"\n{tmc_pool_prompt()}" if task == "TMC" else ""
    parent_text = f"\nBest/parent candidate: {parent}\n" if parent else ""
    if method == "vanilla_mas":
        mode = "Use a direct multi-agent-style discussion internally, then emit one candidate. Do not invent an experiment result."
    elif method == "react":
        mode = "Use ReAct format internally: Thought, Action, and a single candidate action. The runner executes the action after your response."
    elif method == "ai_scientist_v1":
        mode = "Follow AI Scientist v1 archive-guided proposal semantics: analyze the sorted archive, choose exploration or exploitation, then emit one novel candidate inside <candidate> tags."
    elif method == "ai_scientist_v2":
        mode = "Follow AI Scientist v2 best-first tree semantics: refine the best/parent candidate when useful, while preserving diversity and exact schema."
    elif method == "ai_researcher":
        mode = "Follow AI Researcher analysis -> divergent hypotheses -> convergent selection semantics, but return only one final candidate for this evaluator turn."
    else:
        mode = "Follow PiFlow-style principle-aware planning: infer a concise mechanism from the archive, use it to guide one testable candidate, and do not claim an unobserved result."
    analysis_text = "Prior analysis:\n" + analysis if analysis else ""
    return f"""Task: {schema}{pool}
Method profile: {mode}
{parent_text}
Observed archive from this run (do not repeat an evaluated candidate):
{history_text}
{analysis_text}

Return exactly one candidate. Prefer a plain candidate string or JSON with a `candidate` field. Do not return multiple candidates, an outcome, or explanatory text inside the candidate.
"""


def policy_generate(llm: LLM, method: str, task: str, history: list[dict[str, Any]], parent: str | None) -> tuple[str | None, dict[str, Any]]:
    analysis = ""
    call_purposes: list[str]
    if method == "ai_researcher":
        analysis = llm.generate(
            method=method,
            task=task,
            purpose="analysis",
            prompt=build_prompt(method, task, history, parent) + "First analyze patterns only; do not emit a candidate yet.",
        )
        divergent = llm.generate(
            method=method,
            task=task,
            purpose="divergent_generation",
            prompt=build_prompt(method, task, history, parent, analysis) + "Generate several internal alternatives, then label them clearly.",
        )
        selected = llm.generate(
            method=method,
            task=task,
            purpose="convergent_selection",
            prompt=build_prompt(method, task, history, parent, analysis + "\nAlternatives:\n" + divergent) + "Select and emit exactly one final candidate.",
        )
        call_purposes = ["analysis", "divergent_generation", "convergent_selection"]
        response = selected
    elif method == "piflow":
        principle = llm.generate(
            method=method,
            task=task,
            purpose="principle_planning",
            prompt=build_prompt(method, task, history, parent) + "State one concise, testable mechanism before the candidate.",
        )
        response = llm.generate(
            method=method,
            task=task,
            purpose="hypothesis_generation",
            prompt=build_prompt(method, task, history, parent, principle) + "Use the mechanism and emit exactly one final candidate.",
        )
        call_purposes = ["principle_planning", "hypothesis_generation"]
    else:
        response = llm.generate(
            method=method,
            task=task,
            purpose="candidate_generation",
            prompt=build_prompt(method, task, history, parent),
        )
        call_purposes = ["candidate_generation"]
    return extract_candidate(response, task), {"call_purposes": call_purposes, "raw_response": response}


def evaluator(task: str):
    os.environ["PIEVO_SURROGATE_ROOT"] = str(MATCHED_REPO / "surrogates/envs")
    os.environ["PIEVO_CHEMBL_MODEL_PATH"] = str(MATCHED_REPO / "surrogates/envs/ChEMBL35/models/best_r2_model.pt")
    if str(MATCHED_REPO) not in sys.path:
        sys.path.insert(0, str(MATCHED_REPO))
    if task == "MBO":
        from pievo.tools.tool_chembl_official import evaluate_chembl

        return evaluate_chembl
    from pievo.tools.tool_scientific_official import evaluate_nanohelix, evaluate_supercon, evaluate_tmc

    return {"NHO": evaluate_nanohelix, "SPO": evaluate_supercon, "TMC": evaluate_tmc}[task]


def write_manifest(
    output: Path,
    *,
    task: str,
    method: str,
    seed: int,
    budget: int,
    model: str,
    base_url: str,
    status: str,
    exit_code: int | None,
    llm: LLM,
    started_at: str,
    source: dict[str, str | None],
    attempt_count: int,
    primary_count: int,
    applicability_domain: str,
    protocol_path: Path,
    model_track: str | None,
) -> None:
    audit_path = output / AUDIT_NAMES[task]
    invalid = duplicate = primary = 0
    if audit_path.is_file():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(row.get("tool_name", "")).startswith("evaluate_"):
                continue
            if row.get("duplicate_suppressed") is True:
                duplicate += 1
            elif is_primary_result(row):
                primary += 1
            else:
                invalid += 1
    manifest = {
        "schema_version": "baseline_table1_run_manifest.v2",
        "status": status,
        "created_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "run_identity": {
            "task": task,
            "method": method,
            "seed": seed,
            "primary_budget": budget,
            "model": model,
            "model_track": model_track or model_track_for(model),
            "stage": "table1_formal_baseline",
        },
        "runtime": {
            "base_url": base_url,
            "seed_requested": seed,
            "seed_supported_by_provider": llm.seed_supported,
            "llm_calls": llm.call_index,
            "attempt_count": attempt_count,
            "primary_count": primary_count,
            "mu_ref": MU_REF[task],
            "evaluator": "released local PiEvo/PiFlow surrogate adapter",
            "applicability_domain_contract": applicability_domain,
            "generation_config": llm.effective_generation_config(),
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
            "context_messages": llm.protocol["context_messages"],
            "primary_budget_per_cell": llm.protocol.get("primary_budget_per_cell", budget),
        },
        "source_code_hashes": source,
        "provenance": {
            "baseline_reference_root": str(BASELINE_ROOT),
            "matched_evaluator_repo": str(MATCHED_REPO),
            "claim_boundary": "Table 1 matched-evaluator reproduction; not exact hidden paper leaderboard recovery",
            "method_adapter_note": "Legacy reference adapters use an obsolete HTTP API; this run preserves their loop/prompt semantics and binds them to the current released evaluator.",
        },
        "audit": {
            "evaluator_audit": str(audit_path),
            "primary": primary,
            "invalid": invalid,
            "duplicate": duplicate,
            "attempts": attempt_count,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.task not in TASKS or args.method not in METHODS:
        raise SystemExit("unsupported task or method")
    if args.seed < 0 or args.budget < 1:
        raise SystemExit("seed must be non-negative and budget must be positive")
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    if args.temperature is not None and float(args.temperature) != float(protocol["temperature"]):
        raise SystemExit(
            f"controlled protocol requires temperature={protocol['temperature']}; "
            f"received {args.temperature}"
        )
    if args.max_tokens is not None and int(args.max_tokens) != int(protocol["max_tokens"]):
        raise SystemExit(
            f"controlled protocol requires max_tokens={protocol['max_tokens']}; "
            f"received {args.max_tokens}"
        )
    temperature = float(protocol["temperature"] if args.temperature is None else args.temperature)
    max_tokens = int(protocol["max_tokens"] if args.max_tokens is None else args.max_tokens)
    if args.model.lower().startswith("qwen") and protocol.get("enable_thinking") is not False:
        raise SystemExit("controlled Qwen protocol must explicitly disable thinking")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["WORKING_RESULT_DIR"] = str(output)
    os.environ["PIEVO_EXPERIMENT_ROOT"] = str(ROOT)
    os.environ["PIEVO_MAX_PRIMARY_EVALS"] = str(args.budget)
    os.environ["PIEVO_MAX_INVALID_ATTEMPTS"] = str(args.max_invalid_attempts)
    os.environ["PIEVO_STOP_REQUESTED"] = "0"
    os.environ["PIEVO_MBO_FORBIDDEN_CANDIDATES"] = ""
    os.environ["PIEVO_MBO_APPLICABILITY_DOMAIN"] = args.applicability_domain if args.task == "MBO" else "none"
    evaluator_fn = evaluator(args.task)
    api_key = read_api_key(args.api_key_file)
    llm = LLM(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        seed=args.seed,
        output=output,
        temperature=temperature,
        max_tokens=max_tokens,
        protocol=protocol,
    )
    fallback = iter(fallback_candidates(args.task))
    history: list[dict[str, Any]] = []
    best_candidate: str | None = None
    best_output = -math.inf
    primary_count = 0
    attempts = 0
    generation_fallbacks = 0
    forced_candidate: str | None = None
    status = "completed"
    error: str | None = None
    try:
        while primary_count < args.budget and attempts < args.max_attempts:
            attempts += 1
            if forced_candidate is not None:
                candidate = forced_candidate
                forced_candidate = None
                generation = {"call_purposes": [], "fallback_reason": "repair_after_duplicate_or_invalid"}
            else:
                parent = best_candidate if args.method == "ai_scientist_v2" else None
                candidate, generation = policy_generate(llm, args.method, args.task, history, parent)
            if candidate is None:
                candidate = next(fallback, None)
                generation_fallbacks += 1
                generation["fallback_reason"] = "no_candidate_extracted"
            if candidate is None:
                raise RuntimeError("candidate generator and deterministic fallback are exhausted")
            result = evaluator_fn(candidate)
            if result.get("duplicate_suppressed") is True:
                generation["fallback_reason"] = "repair_after_duplicate"
                forced_candidate = next(fallback, None)
                generation_fallbacks += int(forced_candidate is not None)
            elif result.get("success") is not True:
                generation["fallback_reason"] = "repair_after_invalid_or_failed_evaluation"
                forced_candidate = next(fallback, None)
                generation_fallbacks += int(forced_candidate is not None)
            if result.get("record_type") is None:
                if result.get("duplicate_suppressed") is True:
                    result["record_type"] = "duplicate_suppressed"
                elif is_primary_result(result):
                    result["record_type"] = "primary"
            row = {
                "iteration": attempts,
                "candidate": candidate,
                "output": result.get("output"),
                "success": result.get("success") is True,
                "status": result.get("record_type", "unknown"),
                "error": result.get("error"),
                "call_index": result.get("call_index"),
                "generation": {key: value for key, value in generation.items() if key != "raw_response"},
            }
            with (output / "baseline_search.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            history.append(row)
            if is_primary_result(result):
                primary_count += 1
                value = float(result["output"])
                if value > best_output:
                    best_output = value
                    best_candidate = candidate
        if primary_count != args.budget:
            status = "failed"
            error = f"primary budget incomplete: {primary_count}/{args.budget} after {attempts} attempts"
    except Exception as exc:
        status = "failed"
        error = str(exc)
    summary = {
        "schema_version": "baseline_table1_cell_summary.v1",
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "budget": args.budget,
        "status": status,
        "error": error,
        "primary_count": primary_count,
        "attempts": attempts,
        "generation_fallbacks": generation_fallbacks,
        "best_candidate": best_candidate,
        "best_output": best_output if math.isfinite(best_output) else None,
        "sq_percent": (best_output / MU_REF[args.task] * 100.0) if math.isfinite(best_output) else None,
        "llm_calls": llm.call_index,
        "seed_supported_by_provider": llm.seed_supported,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(
        output,
        task=args.task,
        method=args.method,
        seed=args.seed,
        budget=args.budget,
        model=args.model,
        base_url=args.base_url,
        status=status,
        exit_code=0 if status == "completed" else 1,
        llm=llm,
        started_at=started_at,
        source=source_hashes(args.method),
        attempt_count=attempts,
        primary_count=primary_count,
        applicability_domain=args.applicability_domain,
        protocol_path=protocol_path,
        model_track=protocol.get("model_track"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--base-url", default="https://zjuapi.com/v1")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional plaintext key file; omitted uses the configured local Keychain helper.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CONTROLLED_PROTOCOL_PATH)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=96)
    parser.add_argument("--max-invalid-attempts", type=int, default=96)
    parser.add_argument(
        "--applicability-domain",
        choices=("none", "mbo_structural_envelope_v2"),
        default="none",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
