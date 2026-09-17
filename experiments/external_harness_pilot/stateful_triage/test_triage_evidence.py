import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).with_name("triage_evidence.py")
spec = importlib.util.spec_from_file_location("triage_evidence", MODULE)
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def setup_state(tmp_path):
    os.environ["TRIAGE_STATE"] = str(tmp_path / "state.json")
    te.cmd_init(ns())


def add_claim(cid, statement="claim", parent=None):
    te.cmd_claim(ns(id=cid, statement=statement, parent=parent, replace=False))


def add_evidence(cid, eid, source, stance="support", independent=False, heldout=False):
    te.cmd_evidence(ns(
        claim_id=cid,
        id=eid,
        source=source,
        stance=stance,
        independent=independent,
        heldout=heldout,
        reliability="high",
        note="test",
    ))


def test_two_independent_heldout_supports_accept(tmp_path):
    setup_state(tmp_path)
    add_claim("p")
    add_evidence("p", "e1", "channel-a", independent=True, heldout=True)
    add_evidence("p", "e2", "channel-b", independent=True, heldout=True)
    result = te.evaluate_claim(te.load_state(), "p")
    assert result["action"] == "ADMIT"
    assert result["status"] == "accepted"
    assert result["independent_heldout_support_count"] == 2


def test_same_channel_repeat_without_independence_does_not_accept(tmp_path):
    setup_state(tmp_path)
    add_claim("p")
    add_evidence("p", "e1", "channel-a", independent=False, heldout=False)
    add_evidence("p", "e2", "channel-a", independent=False, heldout=False)
    result = te.evaluate_claim(te.load_state(), "p")
    assert result["action"] == "VERIFY"
    assert result["status"] == "provisional"


def test_heldout_contradiction_retracts_parent_and_descendant(tmp_path):
    setup_state(tmp_path)
    add_claim("p")
    add_evidence("p", "e1", "channel-a", independent=True, heldout=True)
    add_evidence("p", "e2", "channel-b", independent=True, heldout=True)
    te.cmd_decide(ns(claim_id="p"))
    add_claim("child", parent="p")
    add_evidence("child", "ce1", "channel-a", independent=True, heldout=True)
    add_evidence("p", "contra", "channel-c", stance="contradict", independent=True, heldout=True)
    te.cmd_decide(ns(claim_id="p"))
    state = te.load_state()
    assert state["claims"]["p"]["status"] == "retracted"
    assert state["claims"]["child"]["status"] == "retracted"


def test_channel_trust_updates(tmp_path):
    setup_state(tmp_path)
    te.cmd_channel(ns(source="surrogate", outcome="agree", weight=1.0))
    te.cmd_channel(ns(source="surrogate", outcome="disagree", weight=2.0))
    ch = te.load_state()["channels"]["surrogate"]
    assert ch["alpha"] == 2.0
    assert ch["beta"] == 3.0
    assert abs(te.trust(ch) - 0.4) < 1e-9
