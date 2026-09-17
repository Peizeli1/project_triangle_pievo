A stateful evidence-admission tool is available at `/tmp/triage-evidence`.

Use it when the task requires accepting, rejecting, revising, or reporting a scientific claim whose support may come from conflicting, dependent, contaminated, or otherwise uncertain evidence. Record the claim and the evidence you actually inspect, including source provenance and whether evidence is independent or held out. Run `/tmp/triage-evidence decide --claim-id <id>` before committing to such a scientific conclusion, and use the returned ADMIT / VERIFY / QUARANTINE decision in your reasoning.

The tool does not replace task execution: still inspect the provided files, make every requested code/config/report change, and run the task's visible validation commands when available. Do not inspect hidden verifier tests, reference solutions, or other forbidden evaluation artifacts.
