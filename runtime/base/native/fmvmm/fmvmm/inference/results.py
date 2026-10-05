from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WaldTestResult:
    statistic: float
    df: int
    pvalue: float
    method: str = "wald"
    reference: str = "chi2"
    warning: Optional[str] = None


@dataclass(frozen=True)
class ScoreTestResult:
    statistic: float
    df: int
    pvalue: float
    method: str = "score"
    reference: str = "chi2"
    warning: Optional[str] = None


@dataclass(frozen=True)
class LRTResult:
    statistic: float
    df: int
    pvalue: float
    ll_full: float
    ll_null: float
    method: str = "lrt"
    reference: str = "chi2"
    warning: Optional[str] = None


@dataclass(frozen=True)
class BootstrapResult:
    statistic: float
    pvalue: float
    bootstrap_statistics: object
    B: int
    method: str = "parametric_bootstrap"
    warning: Optional[str] = None
