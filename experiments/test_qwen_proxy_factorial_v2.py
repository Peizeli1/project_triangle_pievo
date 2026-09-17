from experiments.qwen_required_suite.run_qwen_proxy_factorial_v2 import (
    evaluate,
    parse_selected_x,
    prompt,
)


def test_gate_is_the_only_admission_difference_for_exploit() -> None:
    blind = evaluate(0.8, False)
    gated = evaluate(0.8, True)
    assert blind["primary_proxy"] == gated["primary_proxy"]
    assert blind["heldout_true"] == gated["heldout_true"]
    assert blind["update_allowed"] is True
    assert gated["update_allowed"] is False


def test_afe_selected_x_parser_prefers_explicit_selection() -> None:
    text = '{"A": 0.2, "B": 0.5, "C": 0.9, "selected_x": 0.5}'
    assert parse_selected_x(text) == 0.5


def test_afe_prompt_has_three_roles_without_hidden_objective() -> None:
    history = [evaluate(0.15, True), evaluate(0.45, True), evaluate(0.8, True)]
    text = prompt(history, True)
    assert "local_exploitation" in text
    assert "cross_region" in text
    assert "orthogonal_exploration" in text
    assert "0.32" not in text
