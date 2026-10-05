from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from fmvmm.inference.parameters import (
    alr,
    chol_names,
    covariance_from_info,
    make_parameter_vector,
    vech_names,
    vech_upper,
)
from fmvmm.distributions import multivariate_genhyperbolic as ghypmv
from fmvmm.utils.utils_dist import pack_mvn_unconstrained
from fmvmm.utils.utils_dist import pack_smsn_unconstrained
from fmvmm.utils import utils_dmm


def _as_scalar(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _flatten(items):
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            arr = np.asarray(item, dtype=float)
            out.extend(arr.reshape(-1).tolist())
    return np.asarray(out, dtype=float)


def _dist_short_name(dist_module):
    name = getattr(dist_module, "__name__", str(dist_module)).split(".")[-1]
    mapping = {
        "multivariate_norm": "mvn",
        "multivariate_genhyperbolic": "mghp",
        "multivariate_genskewt": "mgst",
        "multivariate_hyperbolic": "mvhb",
        "multivariate_norminvgauss": "mnig",
        "multivariate_t": "mvt",
        "multivariate_vargamma": "mvvg",
        "multivariate_skew_laplace": "mvsl",
        "multivariate_skew_t_smsn": "mvst",
        "multivariate_skewnorm": "mvsn",
        "multivariate_skewnorm_cont": "msnc",
        "multivariate_skewslash": "mssl",
        "multivariate_slash": "msl",
    }
    return mapping.get(name, name)


def _smsn_names(prefix, p, has_shape=True, has_nu=False, slash_lambda=False):
    names = [f"{prefix}.mu[{i}]" for i in range(p)]
    if has_shape:
        shape_name = "lambda_fixed" if slash_lambda else "lambda"
        names.extend(f"{prefix}.{shape_name}[{i}]" for i in range(p))
    names.extend(vech_names(f"{prefix}.sigma", p))
    if has_nu:
        if has_nu == 2:
            names.extend([f"{prefix}.nu[0]", f"{prefix}.nu[1]"])
        else:
            names.append(f"{prefix}.nu")
    return names


def component_theta_and_names(dist_name: str, alpha, component: int,
                              prefix: str | None = None) -> tuple[np.ndarray, List[str]]:
    prefix = prefix or f"component[{component}].{dist_name}"

    if dist_name == "mvn":
        mu, sigma = alpha
        p = len(mu)
        theta = pack_mvn_unconstrained(p=p, mu=mu, sigma=sigma)
        names = [f"{prefix}.mu[{i}]" for i in range(p)] + chol_names(f"{prefix}.sigma", p)
        return theta, names

    if dist_name == "mghp":
        lmbda, chi, psi, mu, sigma, gamma = alpha
        p = len(mu)
        theta = ghypmv.pack_gh_alpha_bar_unconstrained(
            p=p, lmbda=lmbda, alpha_bar=ghypmv.alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            free=("lmbda", "alpha_bar"),
        )
        names = ([f"{prefix}.lambda", f"{prefix}.log_alpha_bar"] +
                 [f"{prefix}.mu[{i}]" for i in range(p)] +
                 chol_names(f"{prefix}.sigma", p) +
                 [f"{prefix}.gamma[{i}]" for i in range(p)])
        return theta, names

    if dist_name in {"mnig", "mvhb"}:
        chi, psi, mu, sigma, gamma = alpha
        p = len(mu)
        fixed_lambda = -0.5 if dist_name == "mnig" else (p + 1) / 2.0
        theta = ghypmv.pack_gh_alpha_bar_unconstrained(
            p=p, lmbda=fixed_lambda, alpha_bar=ghypmv.alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            free=("alpha_bar",),
        )
        names = ([f"{prefix}.log_alpha_bar"] +
                 [f"{prefix}.mu[{i}]" for i in range(p)] +
                 chol_names(f"{prefix}.sigma", p) +
                 [f"{prefix}.gamma[{i}]" for i in range(p)])
        return theta, names

    if dist_name == "mvvg":
        lmbda, psi, mu, sigma, gamma = alpha
        p = len(mu)
        theta = ghypmv.pack_vg_unconstrained(
            p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma,
        )
        names = ([f"{prefix}.log_lambda"] +
                 [f"{prefix}.mu[{i}]" for i in range(p)] +
                 chol_names(f"{prefix}.sigma", p) +
                 [f"{prefix}.gamma[{i}]" for i in range(p)])
        return theta, names

    if dist_name == "mgst":
        lmbda, chi, mu, sigma, gamma = alpha
        p = len(mu)
        theta = ghypmv.pack_gst_unconstrained(
            p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma,
        )
        names = ([f"{prefix}.log_minus_lambda_minus_one"] +
                 [f"{prefix}.mu[{i}]" for i in range(p)] +
                 chol_names(f"{prefix}.sigma", p) +
                 [f"{prefix}.gamma[{i}]" for i in range(p)])
        return theta, names

    if dist_name in {"mvsn", "mvst", "msnc", "mssl"}:
        mu, sigma, shape, *rest = alpha
        p = len(mu)
        tail = _flatten(rest)
        tail_transforms = {
            "mvsn": (),
            "mvst": ("positive",),
            "mssl": ("positive",),
            "msnc": ("unit", "unit"),
        }[dist_name]
        theta = pack_smsn_unconstrained(
            p=p,
            mu=mu,
            sigma=sigma,
            shape=shape,
            tail=tail,
            tail_transforms=tail_transforms,
        )
        names = (
            [f"{prefix}.mu[{i}]" for i in range(p)]
            + [f"{prefix}.lambda[{i}]" for i in range(p)]
            + chol_names(f"{prefix}.sigma", p)
        )
        if dist_name in {"mvst", "mssl"}:
            names.append(f"{prefix}.log_nu")
        elif dist_name == "msnc":
            names.extend([f"{prefix}.logit_nu[0]", f"{prefix}.logit_nu[1]"])
        return theta, names

    if dist_name in {"mvt", "msl"}:
        mu, sigma, nu = alpha
        p = len(mu)
        theta = pack_smsn_unconstrained(
            p=p,
            mu=mu,
            sigma=sigma,
            tail=np.asarray(nu, dtype=float).reshape(-1),
            tail_transforms=("positive",),
        )
        names = (
            [f"{prefix}.mu[{i}]" for i in range(p)]
            + chol_names(f"{prefix}.sigma", p)
            + [f"{prefix}.log_nu"]
        )
        return theta, names

    if dist_name == "mvsl":
        mu, sigma, gamma = alpha
        p = len(mu)
        theta = pack_smsn_unconstrained(
            p=p, mu=mu, sigma=sigma, shape=gamma
        )
        names = (
            [f"{prefix}.mu[{i}]" for i in range(p)]
            + [f"{prefix}.gamma[{i}]" for i in range(p)]
            + chol_names(f"{prefix}.sigma", p)
        )
        return theta, names

    theta = _flatten(alpha)
    names = [f"{prefix}.theta[{i}]" for i in range(theta.size)]
    return theta, names


@dataclass
class InferenceAdapter:
    model: Any

    def parameter_vector(self, parameterization="internal", info_method="auto"):
        raise NotImplementedError

    def info(self, parameterization="internal", method="auto"):
        pv = self.parameter_vector(parameterization=parameterization, info_method=method)
        return pv.information, pv.covariance

    def score(self, parameterization="internal", method="auto"):
        pv = self.parameter_vector(parameterization=parameterization, info_method=method)
        details = pv.details or {}
        S = details.get("observed_score_matrix")
        if S is None:
            raise NotImplementedError("This model adapter does not expose score vectors.")
        return np.asarray(S, dtype=float).sum(axis=0)

    def log_likelihood(self):
        if hasattr(self.model, "log_likelihood_new"):
            return float(self.model.log_likelihood_new)
        raise NotImplementedError("Model does not expose log_likelihood_new.")


@dataclass
class FMVMMAdapter(InferenceAdapter):
    candidate: int | str = "best"

    def _idx(self):
        if self.candidate == "best":
            return int(getattr(
                self.model, "paper_best_index", np.argmin(self.model.list_bic)
            ))
        return int(self.candidate)

    def _theta_names(self, parameterization):
        idx = self._idx()
        pi = np.asarray(self.model.list_pi[idx], dtype=float)
        alpha = self.model.list_alpha[idx]
        mix = self.model.worked_dist[idx]
        names = []
        theta_parts = []

        if parameterization in {"eta", "internal"}:
            theta_parts.append(alr(pi))
            names.extend(f"eta[{j}]" for j in range(len(pi) - 1))
        else:
            theta_parts.append(pi)
            names.extend(f"pi[{j}]" for j in range(len(pi)))

        for j, dist_module in enumerate(mix):
            dist_name = _dist_short_name(dist_module)
            theta_j, names_j = component_theta_and_names(dist_name, alpha[j], j)
            theta_parts.append(theta_j)
            names.extend(names_j)

        return np.concatenate(theta_parts), names

    def parameter_vector(self, parameterization="internal", info_method="auto"):
        parameterization = "eta" if parameterization == "internal" else parameterization
        info_param = "eta" if parameterization == "eta" else "user"
        info, se, details = self.model.get_info_mat(
            method=info_method,
            parameterization=info_param,
            return_details=True,
        )
        idx = self._idx()
        theta, names = self._theta_names(parameterization)
        cov = details[idx]["cov"]
        return make_parameter_vector(
            theta=theta,
            names=names,
            parameterization=parameterization,
            covariance=cov,
            information=info[idx],
            details=details[idx],
        )

    def log_likelihood(self):
        return float(self.model.list_log_likelihood[self._idx()])


@dataclass
class DMMAdapter(InferenceAdapter):
    def parameter_vector(self, parameterization="internal", info_method="score"):
        pi = np.asarray(self.model.pi_new, dtype=float)
        alpha = np.asarray(self.model.alpha_new, dtype=float)
        K, p = alpha.shape
        info, _se = self.model.get_info_mat(method=info_method)
        cov_internal = covariance_from_info(info)

        if parameterization in {"eta", "internal"}:
            theta = np.concatenate([alr(pi), alpha.reshape(-1)])
            names = [f"eta[{j}]" for j in range(K - 1)]
            names.extend(f"alpha[{j},{m}]" for j in range(K) for m in range(p))
            cov = cov_internal
            information = info
        else:
            from fmvmm.mixtures.FMVMM import transform_information_to_user_scale
            _, cov, _se_user, _T = transform_information_to_user_scale(
                pi, [p] * K, cov_internal
            )
            information = np.linalg.pinv(cov)
            theta = np.concatenate([pi, alpha.reshape(-1)])
            names = [f"pi[{j}]" for j in range(K)]
            names.extend(f"alpha[{j},{m}]" for j in range(K) for m in range(p))

        return make_parameter_vector(theta, names, parameterization, cov, information)

    def score(self, parameterization="internal", method="score"):
        if parameterization not in {"internal", "eta"}:
            raise NotImplementedError("DMM score tests currently use internal eta-alpha coordinates.")
        pi = np.asarray(self.model.pi_new, dtype=float)
        alpha = np.asarray(self.model.alpha_new, dtype=float)
        gamma = np.asarray(self.model.gamma_temp_ar, dtype=float)
        x = np.asarray(self.model.data, dtype=float)
        if getattr(self.model, "EM_type", "Soft") == "Hard":
            gamma = utils_dmm.hard_assignments(gamma)

        U = np.zeros((len(pi) - 1) + alpha.size, dtype=float)
        for i in range(x.shape[0]):
            U += utils_dmm.score_vector_observation(pi, alpha, x[i], gamma[i])
        return U


@dataclass
class GenericMixtureAdapter(InferenceAdapter):
    def parameter_vector(self, parameterization="internal", info_method="auto"):
        if not hasattr(self.model, "get_info_mat"):
            raise NotImplementedError("Model does not expose get_info_mat.")
        info, _se = self.model.get_info_mat()
        cov = covariance_from_info(info)
        theta = np.zeros(info.shape[0], dtype=float)
        names = [f"theta[{i}]" for i in range(info.shape[0])]
        return make_parameter_vector(theta, names, parameterization, cov, info)


def as_adapter(model, **kwargs) -> InferenceAdapter:
    if hasattr(model, "worked_dist") and hasattr(model, "list_alpha"):
        return FMVMMAdapter(model, candidate=kwargs.get("candidate", "best"))
    if hasattr(model, "alpha_new") and np.asarray(getattr(model, "alpha_new")).ndim == 2:
        return DMMAdapter(model)
    return GenericMixtureAdapter(model)
