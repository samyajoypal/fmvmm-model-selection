from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from fmvmm.inference.parameters import ParameterVector


@dataclass(frozen=True)
class LinearConstraint:
    R: np.ndarray
    r: np.ndarray
    name: str = "linear_constraint"

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.R, dtype=float) @ theta - np.asarray(self.r, dtype=float)

    def jacobian(self, theta: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.R, dtype=float)

    @property
    def df(self) -> int:
        return int(np.asarray(self.R).shape[0])

    @classmethod
    def from_indices(cls, q: int, indices: Sequence[int], values,
                     name: str = "fixed_values") -> "LinearConstraint":
        indices = list(indices)
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 1 and len(indices) > 1:
            values = np.repeat(values, len(indices))
        if values.size != len(indices):
            raise ValueError("values must have length 1 or match indices.")
        R = np.zeros((len(indices), q), dtype=float)
        for row, idx in enumerate(indices):
            R[row, int(idx)] = 1.0
        return cls(R=R, r=values, name=name)

    @classmethod
    def from_names(cls, pv: ParameterVector, names: Sequence[str], values,
                   name: str = "fixed_values") -> "LinearConstraint":
        return cls.from_indices(
            q=pv.theta.size,
            indices=[pv.index(n) for n in names],
            values=values,
            name=name,
        )


@dataclass(frozen=True)
class NonlinearConstraint:
    fun: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray]
    df_value: int
    name: str = "nonlinear_constraint"

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.fun(theta), dtype=float).reshape(-1)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.jac(theta), dtype=float)

    @property
    def df(self) -> int:
        return int(self.df_value)


def fixed_value(pv: ParameterVector, name: str, value: float) -> LinearConstraint:
    return LinearConstraint.from_names(
        pv,
        [name],
        [value],
        name=f"{name} = {value}",
    )


def equal(pv: ParameterVector, left: str, right: str) -> LinearConstraint:
    q = pv.theta.size
    R = np.zeros((1, q), dtype=float)
    R[0, pv.index(left)] = 1.0
    R[0, pv.index(right)] = -1.0
    return LinearConstraint(R=R, r=np.zeros(1), name=f"{left} = {right}")


def linear_contrast(pv: ParameterVector, weights: dict[str, float],
                    value: float = 0.0, name: str = "linear_contrast") -> LinearConstraint:
    R = np.zeros((1, pv.theta.size), dtype=float)
    for param_name, weight in weights.items():
        R[0, pv.index(param_name)] = float(weight)
    return LinearConstraint(R=R, r=np.array([value], dtype=float), name=name)
