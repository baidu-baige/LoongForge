import pytest

from loongforge.train.checkpointing import _load_model_state_dict


class _FakeModule:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def load_state_dict(self, state_dict, strict):
        self.calls.append(strict)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("load failed")
        return "loaded"


def test_load_state_dict_success_does_not_fallback():
    module = _FakeModule()
    _load_model_state_dict(module, {}, strict=True)
    assert module.calls == [True]


def test_load_state_dict_strict_fallback_is_preserved():
    module = _FakeModule(failures=1)
    _load_model_state_dict(module, {}, strict=True)
    assert module.calls == [True, False]


def test_load_state_dict_non_strict_error_propagates():
    module = _FakeModule(failures=1)
    with pytest.raises(RuntimeError, match="load failed"):
        _load_model_state_dict(module, {}, strict=False)
    assert module.calls == [False]


def test_load_state_dict_fallback_error_propagates():
    module = _FakeModule(failures=2)
    with pytest.raises(RuntimeError, match="load failed"):
        _load_model_state_dict(module, {}, strict=True)
    assert module.calls == [True, False]
