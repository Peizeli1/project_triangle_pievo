"""Zero-API G1 fairness-control tests."""

from experiments.common_candidate_generator import generate, replay, validate_candidate


def test_common_generator_replays_byte_identically() -> None:
    assert replay(task="MBO", seed=0) == replay(task="MBO", seed=0)
    assert replay(task="MBO", seed=0) != replay(task="MBO", seed=1)


def test_common_generator_outputs_valid_unique_candidates() -> None:
    for seed in (0, 1, 2):
        for slot in range(8):
            values = generate(task="MBO", seed=seed, slot=slot, count=3)
            assert len(values) == len(set(values)) == 3
            assert all(validate_candidate(value) for value in values)

