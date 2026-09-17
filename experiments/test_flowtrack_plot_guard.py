from pathlib import Path

from pievo.group.flowtrack import FlowTracker


def test_disabled_plots_skip_short_trace_plotters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PIEVO_DISABLE_PLOTS", "1")
    tracker = FlowTracker(target_dir=str(tmp_path))
    calls = []

    def unexpected_plot_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("plotting should be skipped when PIEVO_DISABLE_PLOTS=1")

    tracker.plot_uncertainty = unexpected_plot_call
    tracker.plot_principle_survival_stream = unexpected_plot_call
    tracker.compute_all_metrics()

    assert calls == []
