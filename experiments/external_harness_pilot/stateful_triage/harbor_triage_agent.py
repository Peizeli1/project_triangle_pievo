from __future__ import annotations

from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SelfReviewTerminus(Terminus2):
    """Terminus-2 with a task-agnostic final-answer self-review contract."""

    @staticmethod
    def name() -> str:
        return "terminus-2-self-review"

    def version(self) -> str | None:
        return "1.0.0"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        review_contract = """
SELF-REVIEW CONTRACT
Before committing the final response, review the proposed answer and every
artifact you changed against the original task. Check that the conclusion is
supported by evidence actually inspected, that required outputs are complete,
and that no contradictory observation was ignored. Correct any identified
error before finishing. Never inspect hidden verifier tests or reference
solutions.
""".strip()
        await super().run(review_contract + "\n\n" + instruction, environment, context)


class StatefulTriageTerminus(Terminus2):
    """Terminus-2 with a deterministic, persistent evidence-admission CLI installed."""

    TOOL_REMOTE = "/tmp/triage-evidence"
    STATE_REMOTE = "/tmp/triage-evidence-state.json"

    @staticmethod
    def name() -> str:
        return "terminus-2-stateful-triage"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        local_tool = Path(__file__).with_name("triage_evidence.py")
        await environment.upload_file(local_tool, self.TOOL_REMOTE)
        result = await environment.exec(
            f"chmod 755 {self.TOOL_REMOTE}", timeout_sec=10, user="root"
        )
        if result.return_code != 0:
            raise RuntimeError(f"failed to install triage-evidence: {result.stderr}")
        init = await environment.exec(
            f"{self.TOOL_REMOTE} init", timeout_sec=10, user=environment.default_user
        )
        if init.return_code != 0:
            raise RuntimeError(f"failed to initialize triage-evidence: {init.stderr}")
        seed_claim = await environment.exec(
            f"{self.TOOL_REMOTE} claim --id task-conclusion --statement 'Final scientific conclusion for this task'",
            timeout_sec=10,
            user=environment.default_user,
        )
        if seed_claim.return_code != 0:
            raise RuntimeError(f"failed to initialize task claim: {seed_claim.stderr}")
        await super().setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        tool_contract = """
STATEFUL EVIDENCE-ADMISSION COMPONENT
A deterministic stateful tool is installed at /tmp/triage-evidence. The claim
`task-conclusion` has already been created. Before committing to the final
scientific conclusion required by this task, you must record the material
evidence you actually inspected with `triage-evidence evidence`, preserving its
source provenance and marking evidence independent or held out only when that is
true. Then run `/tmp/triage-evidence decide --claim-id task-conclusion` and use
its ADMIT / VERIFY / QUARANTINE result when forming the final conclusion. Use
`/tmp/triage-evidence --help` or subcommand `--help` if needed. This component
does not replace task execution: complete all requested files/code/reports and
visible validation. Never inspect hidden verifier tests or reference solutions.
""".strip()
        await super().run(tool_contract + "\n\n" + instruction, environment, context)
        if await environment.is_file(self.STATE_REMOTE):
            await environment.download_file(
                self.STATE_REMOTE,
                Path(self.logs_dir) / "triage_evidence_state.json",
            )
