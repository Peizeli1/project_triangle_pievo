#!/usr/bin/env python3
import re
from pathlib import Path

tex_path = Path("ai4ai/iclr2027_conference.tex")
tex = tex_path.read_text(encoding="utf-8")

# 1. Check balanced braces
opens = tex.count("{")
closes = tex.count("}")
assert opens == closes, f"Unbalanced braces: {opens} opens vs {closes} closes"
print(f"Braces verified: {opens} open, {closes} close (balanced).")

# 2. Check environments
stack = []
for line_num, line in enumerate(tex.splitlines(), start=1):
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
        action, env = m.group(1), m.group(2)
        if action == "begin":
            stack.append((env, line_num))
        else:
            if not stack:
                raise ValueError(f"Unmatched \\end{{{env}}} at line {line_num}")
            top_env, top_line = stack.pop()
            if top_env != env:
                raise ValueError(f"Mismatched environment at line {line_num}: expected \\end{{{top_env}}} (from line {top_line}), got \\end{{{env}}}")

if stack:
    raise ValueError(f"Unclosed environments remaining: {stack}")
print("Environments verified: all \\begin and \\end blocks match perfectly.")

# 3. Check labels
for target in ["tab:table6_cross_harness", "tab:table7_plugin", "tab:table8_ablation", "sec:sciagentarena"]:
    assert f"\\label{{{target}}}" in tex, f"Missing label: {target}"
    assert f"\\ref{{{target}}}" in tex, f"Missing reference: {target}"
    print(f"Label and reference verified: {target}")

print("\nSUCCESS: All LaTeX syntax, environment, and cross-reference checks passed!")
