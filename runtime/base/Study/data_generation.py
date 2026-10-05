"""Fixed synthetic designs and independent test observations for the study."""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, logsumexp
from fmvmm_engine import fm, probabilities, log_scores, initial_parameters


def parameters(task):
    k, p = int(task.get("true_k", 3)), int(task.get("p", 2))
    separation = float(task.get("separation", 4.))
    if k == 1:
        centers = np.zeros((1,p))
    else:
        angle = np.arange(k)*2*np.pi/k
        centers = np.zeros((k,p))
        centers[:,:2] = np.column_stack([np.cos(angle),np.sin(angle)])*separation/(2*np.sin(np.pi/k))
    proportions = np.repeat(1/k,k)
    if task.get("weights", "balanced") == "unequal" and k > 1:
        proportions = np.geomspace(1., .12, k)
        proportions /= proportions.sum()
    pattern = task.get("truth", "heterogeneous")
    family_pattern = {"heterogeneous":["mvn","mvt","mvsn"], "gaussian":["mvn"],
                      "skew_t":["mvt","mvst","mvsn"], "near_gaussian":["mvn","mvt","mvsn"],
                      "contaminated":["mvn"], "gh":["mnig","mvsn"]}[pattern]
    names = [family_pattern[j%len(family_pattern)] for j in range(k)]
    alphas = []
    for j,name in enumerate(names):
        rho = .25 if j%2 == 0 else -.20
        covariance = rho**np.abs(np.arange(p)[:,None]-np.arange(p)[None,:])
        covariance *= .8 + .2*(j%3)
        shape = np.zeros(p)
        shape[:2] = [.35,.10] if pattern == "near_gaussian" else [4.,1.5]
        df = 40. if pattern == "near_gaussian" else 5.
        if name == "mvn":
            alpha = (centers[j],covariance)
        elif name == "mvt":
            alpha = (centers[j],covariance*(df-2)/df,df)
        elif name in {"mvsn","mvst"}:
            scale = covariance if name == "mvsn" else covariance*(df-2)/df
            val,vec = np.linalg.eigh(scale)
            mean_shift = np.sqrt(2/np.pi)*(vec*np.sqrt(val))@vec.T@shape/np.sqrt(1+shape@shape)
            if name == "mvst":
                mean_shift *= np.sqrt(df/2)*np.exp(gammaln((df-1)/2)-gammaln(df/2))
            alpha = (centers[j]-mean_shift,scale,shape)
            if name == "mvst":
                alpha = (*alpha,np.array([df]))
        else:
            alpha = initial_parameters(name,centers[j],covariance)
        alphas.append(alpha)
    return {"pi":proportions,"families":names,"alpha":alphas,"centers":centers,
            "contamination":.05 if pattern == "contaminated" else 0.}


def sample(n, truth, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(int(seed)%2**32)
    labels = rng.choice(len(truth["pi"]),size=n,p=truth["pi"])
    p = truth["centers"].shape[1]
    x = np.empty((n,p))
    for j,(name,alpha) in enumerate(zip(truth["families"],truth["alpha"])):
        index = np.flatnonzero(labels==j)
        if not len(index):
            continue
        x[index] = np.asarray(fm.dist_map[name].rvs(*alpha,size=len(index))).reshape(len(index),p)
        if truth["contamination"]:
            contaminated = rng.random(len(index)) < truth["contamination"]
            x[index[contaminated]] = rng.multivariate_normal(alpha[0],25*np.asarray(alpha[1]),size=contaminated.sum())
    if not np.isfinite(x).all():
        raise ValueError("Non-finite generated data")
    return x,labels


def true_posterior(x, truth):
    if not truth["contamination"]:
        return probabilities(log_scores(x,truth["families"],truth["pi"],truth["alpha"]))
    a=[]
    for weight,alpha in zip(truth["pi"],truth["alpha"]):
        clean=fm.dist_map["mvn"].logpdf(x,*alpha)+np.log(.95)
        dirty=fm.dist_map["mvn"].logpdf(x,alpha[0],25*np.asarray(alpha[1]))+np.log(.05)
        a.append(np.log(weight)+np.logaddexp(clean,dirty))
    return probabilities(np.column_stack(a))


def generate(task):
    truth=parameters(task)
    seed=int(task.get("data_seed",task["seed"]))
    x,labels=sample(int(task["n"]),truth,seed)
    test,test_labels=sample(int(task.get("test_n",2000)),truth,(seed+50000017)%2**32)
    return {"Xtrain":x,"Xtest":test,"ytrain":labels,"ytest":test_labels,
            "qtrue":true_posterior(x,truth),"qtrue_test":true_posterior(test,truth),
            "metadata":{"truth":truth,"training_seed":seed,"test_seed":(seed+50000017)%2**32,
                        "feature_space":"original simulated coordinates","test_observations_independent":True}}
