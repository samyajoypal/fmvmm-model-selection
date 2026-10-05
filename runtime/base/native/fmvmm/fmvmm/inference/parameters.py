from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


_EPS = 1e-15


@dataclass(frozen=True)
class Parameter:
    name: str
    index: int
    block: str = ""
    component: int | None = None
    scale: str = "internal"


@dataclass(frozen=True)
class ParameterVector:
    theta: np.ndarray
    parameters: List[Parameter]
    parameterization: str
    covariance: np.ndarray | None = None
    information: np.ndarray | None = None
    details: dict | None = None

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def name_to_index(self) -> Dict[str, int]:
        return {p.name: p.index for p in self.parameters}

    def index(self, name: str) -> int:
        try:
            return self.name_to_index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown parameter name: {name}") from exc


def covariance_from_info(info: np.ndarray, use_pinv: bool = True) -> np.ndarray:
    info = np.asarray(info, dtype=float)
    try:
        return np.linalg.pinv(info) if use_pinv else np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(info)


def alr(pi: Sequence[float]) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()
    if pi.size <= 1:
        return np.zeros(0)
    return np.log(pi[:-1] / pi[-1])


def vech_upper(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]


def vech_names(prefix: str, p: int) -> List[str]:
    return [f"{prefix}[{i},{j}]" for i in range(p) for j in range(i, p)]


def chol_names(prefix: str, p: int) -> List[str]:
    names = [f"{prefix}.chol_logdiag[{i}]" for i in range(p)]
    names.extend(f"{prefix}.chol_off[{i},{j}]" for i in range(1, p) for j in range(i))
    return names


def named_parameters(names: Iterable[str], scale: str = "internal") -> List[Parameter]:
    return [Parameter(name=name, index=i, scale=scale) for i, name in enumerate(names)]


def make_parameter_vector(theta, names, parameterization, covariance=None,
                          information=None, details=None) -> ParameterVector:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if len(names) != theta.size:
        raise ValueError(f"Expected {theta.size} names, got {len(names)}.")
    return ParameterVector(
        theta=theta,
        parameters=named_parameters(names, scale=parameterization),
        parameterization=parameterization,
        covariance=covariance,
        information=information,
        details=details,
    )
