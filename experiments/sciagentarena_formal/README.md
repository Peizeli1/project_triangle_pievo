# SciAgentArena formal Table 6--8 run

This directory holds the frozen protocol and runners for the 79-task
SciAgentArena Drug Discovery evaluation. The upstream checkout is pinned at
`official_sources/SciAgentArena` commit
`9865bb0c261bd9a59ef23576805b268b458b59d2`.

The public upstream GitHub release is used directly. The gated Hugging Face
Terminal-Bench conversion is not required and no access control is bypassed.

All methods share the same model, decoding parameters, task inputs, official
scorers, and three seeds. Generation processes cannot read scorer data,
ground-truth files, or another method's output. Official scoring happens only
after a method has committed its candidate program.

The former AutoResearchEval/AARRI placeholders and controlled two-case results
must not be mixed into these tables.
