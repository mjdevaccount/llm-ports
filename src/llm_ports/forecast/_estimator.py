"""The scikit-learn estimator parameter protocol, standard library only.

`Params` gives a class `get_params` / `set_params` read from its `__init__` signature, the
contract `sklearn.base.BaseEstimator` defines, so `sklearn.base.clone`, grid search and pipelines
accept it. It is reimplemented here because this package imports no third-party library.

Invariants:
    - A subclass stores every `__init__` parameter unchanged under the same attribute name.
    - Fitted state lives in attributes ending in `_`, set only by `fit`.

Rationale: ADR-0013.
"""

from __future__ import annotations

import inspect
from typing import Any, Self


class Params:
    """Mixin: `get_params` / `set_params` over the `__init__` parameters."""

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [
            name
            for name, p in sig.parameters.items()
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters by name.

        Args:
            deep: Accepted for the scikit-learn contract; these estimators hold no sub-estimators.

        Returns:
            `name -> value` for every `__init__` parameter. Never raises.
        """
        del deep
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        """Set constructor parameters by name; fitted state is untouched until the next `fit`.

        Returns:
            The estimator.

        Raises:
            ValueError: For a name that is not an `__init__` parameter (scikit-learn's contract).
        """
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self


__all__ = ["Params"]
