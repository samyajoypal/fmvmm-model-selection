"""Regression checks for the multivariate skew-Laplace helpers."""

import numpy as np

from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.mixtures.skewlaplacemix import estimate_alphas_skewlaplace


def test_loglike_and_moments_match_mvsl_parameterization():
    mu = np.array([0.3, -0.4])
    sigma = np.array([[1.1, 0.2], [0.2, 0.7]])
    gamma = np.array([0.35, -0.2])
    x = np.array([[0.0, 0.2], [1.2, -0.7], [-0.5, 0.4]])

    assert np.allclose(mvsl.loglike(x, mu, sigma, gamma), np.sum(mvsl.logpdf(x, mu, sigma, gamma)))
    assert np.allclose(mvsl.mean(mu, sigma, gamma), mu + 3.0 * gamma)
    assert np.allclose(
        mvsl.var(mu, sigma, gamma),
        3.0 * (sigma + 2.0 * np.outer(gamma, gamma)),
    )


def test_rvs_samples_from_own_density_moment_parameterization():
    mu = np.array([0.3, -0.4])
    sigma = np.array([[1.1, 0.2], [0.2, 0.7]])
    gamma = np.array([0.35, -0.2])

    x = mvsl.rvs(mu, sigma, gamma, size=120000, random_state=1234)

    assert np.allclose(x.mean(axis=0), mvsl.mean(mu, sigma, gamma), atol=0.035)
    assert np.allclose(np.cov(x, rowvar=False, bias=True), mvsl.var(mu, sigma, gamma), atol=0.08)


def test_skew_laplace_mstep_uses_unrestricted_scatter_update():
    x = np.array([
        [-1.0, 0.1],
        [-0.4, 0.3],
        [0.2, -0.1],
        [0.7, 0.9],
        [1.4, 0.5],
    ])
    weights = np.array([0.85, 0.75, 0.55, 0.35, 0.20])
    gamma_matrix = weights[:, None]
    alpha_old = [(
        np.array([0.05, 0.15]),
        np.array([[1.2, 0.25], [0.25, 0.9]]),
        np.array([0.18, -0.12]),
    )]

    mu_new, sigma_new, gamma_new = estimate_alphas_skewlaplace(x, gamma_matrix, alpha_old)[0]

    mu_old, sigma_old, gamma_old = alpha_old[0]
    inv_sigma = np.linalg.inv(sigma_old)
    alpha = np.sqrt(1.0 + gamma_old @ inv_sigma @ gamma_old)
    diff_old = x - mu_old
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff_old, inv_sigma, diff_old))
    v1 = alpha / dist
    v2 = (1.0 + alpha * dist) / (alpha * alpha)

    n_eff = weights.sum()
    sum_z_v1 = np.dot(weights, v1)
    sum_z_v2 = np.dot(weights, v2)
    sum_z_x = np.sum(weights[:, None] * x, axis=0)
    sum_z_v1_x = np.sum((weights * v1)[:, None] * x, axis=0)
    expected_gamma = (
        sum_z_x - n_eff * (sum_z_v1_x / sum_z_v1)
    ) / (sum_z_v2 - n_eff * n_eff / sum_z_v1)
    expected_mu = (sum_z_v1_x - n_eff * expected_gamma) / sum_z_v1

    residual = x - expected_mu
    scatter = np.einsum("i,ij,ik->jk", weights * v1, residual, residual)
    cross = np.einsum("i,ij,k->jk", weights, residual, expected_gamma)
    expected_sigma = (
        scatter
        - cross
        - cross.T
        + sum_z_v2 * np.outer(expected_gamma, expected_gamma)
    ) / n_eff

    assert np.allclose(mu_new, expected_mu)
    assert np.allclose(gamma_new, expected_gamma)
    assert np.allclose(sigma_new, 0.5 * (expected_sigma + expected_sigma.T))


if __name__ == "__main__":
    test_loglike_and_moments_match_mvsl_parameterization()
    test_rvs_samples_from_own_density_moment_parameterization()
    test_skew_laplace_mstep_uses_unrestricted_scatter_update()
    print("skew-Laplace distribution tests passed")
