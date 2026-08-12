from torch import nn

from regnav.train import _maybe_compile


def test_maybe_compile_uses_reduce_overhead(monkeypatch):
    model = nn.Linear(2, 1)
    compiled = nn.Linear(2, 1)
    called = {}

    def fake_compile(candidate, *, mode):
        called.update(candidate=candidate, mode=mode)
        return compiled

    monkeypatch.setattr("regnav.train.torch.compile", fake_compile)

    assert _maybe_compile(model, False) is model
    assert _maybe_compile(model, True) is compiled
    assert called == {"candidate": model, "mode": "reduce-overhead"}
