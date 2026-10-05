import numpy as np
from scipy.stats import gamma, norm, invgamma, invgauss
from scipy.optimize import root_scalar

def gig_y_gfn(y, m, beta, lambda_):
    y2 = y * y
    g = 0.5 * beta * y2 * y
    g -= y2 * (0.5 * beta * m + lambda_ + 1.0)
    g += y * ((lambda_ - 1.0) * m - 0.5 * beta) + 0.5 * beta * m
    return g

def rinvgauss(mu, lambda_):
    y = norm.rvs()
    y *= y
    mu2 = mu * mu
    l2 = 2.0 * lambda_
    x1 = mu + mu2 * y / l2 - (mu / l2) * np.sqrt(4.0 * mu * lambda_ * y + mu2 * y * y)

    u = np.random.uniform()
    if u <= mu / (mu + x1):
        return x1
    else:
        return mu2 / x1

def zeroin_gig(ax, bx, f, tol, m, beta, lambda_):
    a, b, c = ax, bx, ax
    fa, fb, fc = f(a, m, beta, lambda_), f(b, m, beta, lambda_), f(c, m, beta, lambda_)

    for _ in range(1000):  # Maximum iterations to avoid infinite loops
        prev_step = b - a
        tol_act = 2.0 * np.sqrt(np.finfo(float).eps) * np.abs(b) + tol / 2.0
        new_step = (c - b) / 2.0

        if np.abs(new_step) <= tol_act or fb == 0:
            return b

        if np.abs(prev_step) >= tol_act and np.abs(fa) > np.abs(fb):
            cb = c - b

            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            if p < (0.75 * cb * q - np.abs(tol_act * q) / 2.0) and p < np.abs(prev_step * q / 2.0):
                new_step = p / q

        if np.abs(new_step) < tol_act:
            if new_step > 0:
                new_step = tol_act
            else:
                new_step = -tol_act

        a, fa = b, fb
        b += new_step
        fb = f(b, m, beta, lambda_)

        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa

def rgig(n, lambda_, chi, psi):
    ZTOL = np.sqrt(np.finfo(float).eps)
    samps = np.empty(n)

    if chi < ZTOL and lambda_ > 0.0:
        samps = gamma.rvs(lambda_, scale=2.0/psi, size=n)
        return samps

    if psi < ZTOL and lambda_ < 0.0:
        samps = 1.0 / gamma.rvs(0.0 - lambda_, scale=2.0/chi, size=n)
        return samps

    if lambda_ == -0.5:
        alpha = np.sqrt(chi / psi)
        samps = invgauss.rvs(alpha, scale=chi, size=n)
        return samps

    alpha = np.sqrt(chi / psi)
    beta2 = psi * chi
    beta = np.sqrt(psi * chi)
    lm1 = lambda_ - 1.0
    lm12 = lm1 * lm1
    m = (lm1 + np.sqrt(lm12 + beta2)) / beta
    m1 = m + 1.0/m

    upper = m
    while gig_y_gfn(upper, m, beta, lambda_) <= 0:
        upper *= 2.0

    yM = zeroin_gig(0.0, m, gig_y_gfn, ZTOL, m, beta, lambda_)
    yP = zeroin_gig(m, upper, gig_y_gfn, ZTOL, m, beta, lambda_)

    a = (yP - m) * (yP/m) ** (0.5 * lm1)
    a *= np.exp(-0.25 * beta * (yP + 1.0/yP - m1))
    b = (yM - m) * (yM/m) ** (0.5 * lm1)
    b *= np.exp(-0.25 * beta * (yM + 1.0/yM - m1))
    c = -0.25 * beta * m1 + 0.5 * lm1 * np.log(m)

    need = 1
    for i in range(n):
        while need:
            R1, R2 = np.random.uniform(size=2)
            Y = m + a * R2 / R1 + b * (1.0 - R2) / R1
            if Y > 0.0:
                if -np.log(R1) >= -0.5 * lm1 * np.log(Y) + 0.25 * beta * (Y + 1.0/Y) + c:
                    need = 0
        samps[i] = Y * alpha
        need = 1

    return samps
