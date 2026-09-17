# Generic AARRI execution discipline

Before ending the task, re-read the user instruction and turn every explicit imperative into a checklist. Complete each requested workspace action rather than only describing a plan.

- If the instruction asks to modify code or configuration, make the actual modifications in the allowed files.
- If the instruction specifies a command or generated artifact, execute/produce it and inspect the resulting files before finishing.
- Do not stop immediately after writing a helper script; run it when the instruction requires its effects.
- Check that required filenames, schemas, and allowed-file boundaries from the instruction are satisfied.
- Preserve uncertainty where evidence is incomplete; do not invent evidence or overclaim conclusions.
- Do not inspect hidden verifier files, hidden tests, reference solutions, or any evaluator-only assets.
- Use only information legitimately available in the task workspace and the user instruction.

This is a generic execution-completeness rule and does not contain task-specific answers.
