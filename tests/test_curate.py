from screex.core import curate
from screex.core.index import ScreenState


def _states(n):
    return [ScreenState(i, float(i), float(i + 1), f"{i}_t.png", f"{i}.png") for i in range(n)]


def test_score_states_ranks_by_text_change_and_sharpness():
    states = _states(3)
    states[0].text_added = ["a"]                       # small change
    states[1].text_added = ["a", "b", "c", "d"]        # big change
    states[2].text_added = ["a"]
    curate.score_states(states, sharpness=[0.0, 0.0, 100.0])
    # State 1 wins on text change; state 2 gets the sharpness mass.
    assert states[1].salience > states[0].salience
    assert states[2].salience > states[0].salience


def test_score_states_event_bonus():
    states = _states(2)
    states[1].event = {"type": "error"}
    curate.score_states(states, sharpness=[0.0, 0.0])  # no text/sharpness signal
    assert states[1].salience > states[0].salience     # only the event distinguishes them


def test_score_states_without_sharpness_uses_text_only():
    states = _states(2)
    states[1].text_added = ["x", "y"]
    curate.score_states(states)                        # sharpness omitted
    assert states[1].salience > states[0].salience


def test_select_curated_caps_at_budget_and_spreads():
    states = _states(5)
    # Three equally-salient states (0, 1, 4); with the salience tied, the temporal-coverage
    # bonus should break it toward a spread pick (0 and the far 4), not the adjacent 0 and 1.
    for s in states:
        s.salience = 0.1
    states[0].salience = states[1].salience = states[4].salience = 0.9
    picked = curate.select_curated(states, budget=2)
    assert len(picked) == 2
    idxs = [p["idx"] for p in picked]
    assert idxs == sorted(idxs)                        # ordered by t_start
    assert idxs == [0, 4]                              # spread, not the adjacent 0 & 1


def test_select_curated_budget_bounds():
    states = _states(3)
    curate.score_states(states, [1.0, 2.0, 3.0])
    assert curate.select_curated(states, 0) == []
    assert len(curate.select_curated(states, 99)) == 3  # budget >= len -> all
    assert {p["idx"] for p in curate.select_curated(states, 99)} == {0, 1, 2}


def test_select_curated_empty():
    assert curate.select_curated([], 3) == []


class _StubEmbedder:
    """Deterministic image+text embedder for query-conditioned curation tests."""

    def __init__(self, text_vec, image_vecs):
        self._t, self._i = text_vec, image_vecs

    def embed_text(self, text):
        return self._t

    def embed_image(self, path):
        from pathlib import Path
        return self._i[Path(path).name]


def test_select_curated_for_query_relevance_and_diversity():
    states = _states(4)  # keyframes "0.png".."3.png"
    images = {"0.png": [1.0, 0.0], "1.png": [1.0, 0.0],
              "2.png": [0.0, 1.0], "3.png": [0.2, 0.2]}
    emb = _StubEmbedder([1.0, 0.0], images)
    picks = curate.select_curated_for_query(states, 2, "/base", "q", emb, diversity=0.9)
    idxs = {p["idx"] for p in picks}
    # picks the relevant frame 0, then the diverse frame 2 — NOT the near-duplicate 1
    assert 0 in idxs and 2 in idxs and 1 not in idxs


def test_select_curated_for_query_budget_ge_n_returns_all():
    states = _states(2)
    picks = curate.select_curated_for_query(states, 5, "/base", "q", embedder=None)
    assert len(picks) == 2


def test_clip_embedder_available_or_skipped():
    import pytest
    pytest.importorskip("sentence_transformers")
    emb = curate.ClipEmbedder()
    assert len(emb.embed_text("a settings dialog")) > 0
