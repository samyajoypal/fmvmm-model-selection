"""Screening study adapter. The reference FMVMM source is imported unchanged.

Every fitting/profile operation has a process-level deadline. A private SQLite cache
per dataset permits resumption without creating one shared-file entry per fit.
No cached numerical optimum is described as a certified global optimum.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fmvmm-study-mpl")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
REFERENCE_ROOT = Path(os.environ["FMVMM_REFERENCE_ROOT"]).resolve()
sys.path.insert(0, str(REFERENCE_ROOT))
import copy
import hashlib
import importlib
import itertools
import json
import multiprocessing as mp
import sqlite3
import time
import traceback
import zlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp, kv, gammaln
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

fm = importlib.import_module("fmvmm.mixtures.FMVMM")
from fmvmm.utils.utils_fmm import fmm_bic

LIBRARIES = {"L3": ["mvn", "mvt", "mvsn"],
             "L5": ["mvn", "mvt", "mvsn", "mvst", "msnc"],
             "L13": list(fm.all_dist)}
DEFAULTS = {"max_iter": 160, "tol": 1e-7, "mstep_max_iter": 60,
            "mstep_maxfun": 1000, "gh_nit": 15, "profile_max_iter": 120,
            "profile_maxfun": 2000, "profile_gh_nit": 60,
            "polish_max_iter": 300, "polish_tol": 1e-9,
            "fit_timeout": 1200, "profile_timeout": 180,
            "pilot_starts": 1, "kmeans_starts": 20,
            "audit": False, "test_size": 2000}


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [native(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(native(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(native(value), sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, path, signature):
        self.folder = Path(path).parent
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        previous = self.get("signature")
        if previous is not None and previous != signature:
            raise ValueError("Refusing to reuse results with changed data, source or configuration")
        self.put("signature", signature)
        self.hits = 0

    def get(self, key):
        row = self.db.execute("SELECT value FROM entries WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(zlib.decompress(row[0]))

    def put(self, key, value):
        packed = zlib.compress(json.dumps(native(value), allow_nan=False).encode(), 6)
        self.db.execute("INSERT OR REPLACE INTO entries VALUES (?,?)", (key, packed))
        self.db.commit()

    def call(self, key, operation, args, timeout):
        value = self.get(key)
        if value is not None:
            self.hits += 1
            return value
        value = bounded(operation, args, timeout)
        self.put(key, value)
        write_json(self.folder / "progress.json", {"last_completed_operation": key, "updated_unix": time.time(),
                                                   "finite": value.get("finite"), "cache_hits": self.hits})
        return value

    def close(self):
        self.db.close()


def bounded(function, args, timeout):
    """Fork only inside a one-CPU worker; terminate timed-out child processes."""
    started = time.perf_counter()
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child():
        receiver.close()
        try:
            sender.send({"ok": True, "value": native(function(*args))})
        except BaseException:
            sender.send({"ok": False, "error": traceback.format_exc()})
        finally:
            sender.close()

    process = context.Process(target=child)
    process.start()
    sender.close()
    packet = None
    try:
        if receiver.poll(timeout):
            try:
                packet = receiver.recv()
            except EOFError:
                pass
    finally:
        receiver.close()
        if packet is None:
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
    if packet and packet["ok"]:
        value = packet["value"]
        value["seconds"] = time.perf_counter() - started
        return value
    return {"finite": False, "status": "error" if packet else "timeout_or_child_exit",
            "error": packet.get("error") if packet else "Operation exceeded its deadline or child exited without a result",
            "seconds": time.perf_counter() - started, "hit_cap": False}


def alpha_arrays(values):
    return [tuple(np.asarray(part, dtype=float) for part in component) for component in values]


def log_scores(x, names, pi, alpha):
    return np.column_stack([np.asarray(fm.dist_map[name].logpdf(x, *params)).reshape(-1) + np.log(weight)
                            for name, weight, params in zip(names, pi, alpha)])


def fitted_scores(x, row):
    return log_scores(x, row["assignment"], row["pi"], alpha_arrays(row["alpha"]))


def probabilities(a):
    return np.exp(a - logsumexp(a, axis=1, keepdims=True))


def fit_mixture(x, names, pi0, alpha0, config):
    names = tuple(names)
    modules = tuple(fm.dist_map[name] for name in names)
    model = fm.fmvmm(n_clusters=len(names), list_of_dist=list(names), candidate_combinations=[names],
                     em_type="soft", max_iter=config["max_iter"], tol=config["tol"], verbose=False,
                     print_log_likelihood=False, n_jobs=1)
    pi, alpha, returned_ll, returned_logq = model._fit(
        x, np.asarray(pi0, dtype=float), copy.deepcopy(alpha_arrays(alpha0)), fm.fmm_estimate_alphas_soft,
        dist_comb=modules, soft_dist_comb=modules,
        smsn_mstep_kwargs={"maxiter": config["mstep_max_iter"], "maxfun": config["mstep_maxfun"],
                           "ftol": 1e-10, "gtol": 1e-5},
        gh_mstep_kwargs={"nit": config["gh_nit"], "reltol": 1e-8, "silent": True,
                         "opt_maxiter": config["mstep_max_iter"], "opt_maxfun": config["mstep_maxfun"]},
        mstep_accept_tol=1e-8)
    a = log_scores(x, names, pi, alpha)
    ll = float(logsumexp(a, axis=1).sum())
    bic = float(fmm_bic(alpha, ll, modules, len(x)))
    q = probabilities(a)
    trace = np.asarray(model.log_likelihoods)
    gain = float(abs(trace[-1] - trace[-2]) / max(abs(trace[-2]), 1e-300)) if len(trace) > 1 else None
    covariance_checks = []
    for name, parameters in zip(names, alpha):
        index = {"mghp": 4, "mgst": 3, "mnig": 3, "mvhb": 3, "mvvg": 3}.get(name, 1)
        covariance_checks.append(float(np.linalg.eigvalsh(np.asarray(parameters[index])).min()))
    finite = bool(np.isfinite(a).all() and np.isfinite(trace).all() and np.isfinite(bic) and np.all(np.asarray(pi) > 0)
                  and np.allclose(q.sum(1), 1) and np.isclose(np.sum(pi), 1)
                  and np.isclose(ll, returned_ll, atol=1e-7, rtol=1e-10)
                  and np.allclose(q, np.exp(returned_logq), atol=1e-9, rtol=1e-8))
    return {"finite": finite, "assignment": names, "structure": sorted(names), "pi": pi, "alpha": alpha,
            "bic": bic, "loglik": ll, "trace": trace, "iterations": max(0, len(trace)-1),
            "hit_cap": len(trace) >= config["max_iter"], "last_relative_gain": gain,
            "monotone": bool(np.all(np.diff(trace) >= -1e-7)),
            "minimum_covariance_eigenvalue": min(covariance_checks), "minimum_weight": float(np.min(pi)),
            "near_degenerate": bool(min(covariance_checks) < 1e-5 or np.min(pi) < 1e-3),
            "optimizer_success_exposed": False}


def pilot_fit(x, k, seed, config):
    labels = KMeans(n_clusters=k, n_init=config["kmeans_starts"], random_state=seed).fit_predict(x)
    weights = np.eye(k)[labels]
    alphas = [fm._weighted_mvn_update(x, weights[:, j], None) for j in range(k)]
    result = fit_mixture(x, ("mvn",)*k, weights.mean(0), alphas, config)
    result.update({"geometric_starts": config["kmeans_starts"], "initial_component_fits": k})
    return result


def initial_parameters(name, mean, covariance, shape=None, tail=5.):
    p = len(mean)
    shape = np.zeros(p) if shape is None else shape
    if name == "mvn":
        return mean, covariance
    if name == "mvt":
        return mean, covariance*(tail-2)/tail, np.array([tail])
    if name in {"mvsn", "mvst", "mssl", "msnc"}:
        delta = shape / np.sqrt(1 + shape @ shape)
        scale = covariance / (1 - 2/np.pi*(delta @ delta))
        val, vec = np.linalg.eigh(scale)
        location = mean - np.sqrt(2/np.pi)*(vec*np.sqrt(val))@vec.T@delta
        base = (location, scale, shape)
        return base if name == "mvsn" else (*base, np.array([.10, .20]) if name == "msnc" else np.array([tail]))
    if name == "msl":
        return mean, covariance * .75, np.array([4.])
    if name == "mvsl":
        return mean, covariance, np.zeros(p)
    if name == "mgst":
        return -2.5, 3., mean, covariance, np.zeros(p)
    if name == "mvvg":
        return 3., 6., mean, covariance, np.zeros(p)
    order = 1. if name == "mghp" else -.5 if name == "mnig" else (p+1)/2
    chi, psi = fm.mghp._alphabar2chipsi(1., order)
    expected_w = np.sqrt(chi/psi)*kv(order+1, np.sqrt(chi*psi))/kv(order, np.sqrt(chi*psi))
    base = (chi, psi, mean, covariance/expected_w, np.zeros(p))
    return (order, *base) if name == "mghp" else base


def profile_cell(x, weights, name, config):
    mean, covariance = fm._weighted_mvn_update(x, weights, None)
    standardized = (x-mean)/np.sqrt(np.diag(covariance))
    direction = (weights[:, None]*standardized**3).sum(0)/weights.sum()
    direction /= max(np.linalg.norm(direction), 1e-12)
    if np.linalg.norm(direction) < .5:
        direction = np.eye(x.shape[1])[0]
    starts = [initial_parameters(name, mean, covariance)]
    if name == "mvt":
        starts = [initial_parameters(name, mean, covariance, tail=df) for df in (4., 20.)]
    elif name in {"mvsn", "mvst", "mssl", "msnc"}:
        starts += [initial_parameters(name, mean, covariance, sign*2*direction) for sign in (1., -1.)]
        if name == "mvsn":
            group = x[weights > .5]
            if len(group) > x.shape[1]+2:
                try:
                    starts.append(fm.dist_map[name].fit(group, maxiter=40, ftol=1e-7))
                except Exception:
                    pass
    best, value = None, -np.inf
    records = []
    for initial in starts:
        old = fm._weighted_component_loglik(name, x, weights, initial)
        if np.isfinite(old) and old > value:
            best, value = initial, old
        try:
            candidate = fm._soft_update_one_component(
                name, x, weights, copy.deepcopy(initial),
                smsn_mstep_kwargs={"maxiter": config["profile_max_iter"], "maxfun": config["profile_maxfun"], "ftol": 1e-10},
                gh_mstep_kwargs={"nit": config["profile_gh_nit"], "reltol": 1e-8, "silent": True,
                                 "opt_maxiter": config["profile_max_iter"], "opt_maxfun": config["profile_maxfun"]})
            new = fm._weighted_component_loglik(name, x, weights, candidate)
            records.append({"initial": old, "final": new, "finite": np.isfinite(new)})
            if np.isfinite(new) and new > value:
                best, value = candidate, new
        except Exception as exc:
            records.append({"initial": old, "finite": False, "error": repr(exc)})
    if best is None:
        return {"finite": False, "starts": records}
    dimension = int(fm.dist_map[name].total_params(*best))
    return {"finite": True, "alpha": best, "weighted_loglik": value, "dimension": dimension,
            "score": 2*value - dimension*np.log(len(x)), "starts": records,
            "numerical_starts": len(starts) if name != "mvn" else 0,
            "exact_profiles": int(name == "mvn"), "global_profile_certified": False}


def shortlists(scores, library, n):
    """Enumerate score tuples only; mixture fits occur after lists are frozen."""
    k, ell = scores.shape
    if not np.isfinite(scores).all():
        raise RuntimeError("A profile cell failed; do not silently remove its family")
    deficits = scores.max(1, keepdims=True)-scores
    orders = list(itertools.product(range(ell), repeat=k))
    costs = np.array([sum(deficits[j, m[j]] for j in range(k)) for m in orders])
    choices = {"leading": [tuple(scores.argmax(1))],
               "joint_1logn": [m for m,c in zip(orders,costs) if c <= np.log(n)+1e-10],
               "joint_2logn": [m for m,c in zip(orders,costs) if c <= 2*np.log(n)+1e-10],
               "joint_4logn": [m for m,c in zip(orders,costs) if c <= 4*np.log(n)+1e-10],
               "joint_6logn": [m for m,c in zip(orders,costs) if c <= 6*np.log(n)+1e-10],
               "top_two": [m for m in orders if all(m[j] in np.argsort(-scores[j], kind="stable")[:2] for j in range(k))],
               "cartesian_2logn": [m for m in orders if all(deficits[j,m[j]] <= 2*np.log(n)+1e-10 for j in range(k))]}
    ordered = {rule: [tuple(library[l] for l in m) for m in values] for rule,values in choices.items()}
    canonical = {rule: sorted({tuple(sorted(m)) for m in values}) for rule,values in ordered.items()}
    return canonical, ordered, deficits


def aligned_error(q, target):
    if q.shape[1] != target.shape[1]:
        return None
    cost = np.array([[np.abs(q[:,j]-target[:,h]).mean() for h in range(q.shape[1])] for j in range(q.shape[1])])
    rows, cols = linear_sum_assignment(cost)
    return float(.5*cost[rows,cols].sum())


def metrics(row, x, test, labels=None, test_labels=None, qtrue=None, qtrue_test=None):
    a = fitted_scores(x, row)
    q = probabilities(a)
    result = {"bic": row["bic"], "winner": row["structure"], "winning_assignment": row["assignment"],
              "posterior_tv_train": aligned_error(q, qtrue) if qtrue is not None else None,
              "ari_train": float(adjusted_rand_score(labels, q.argmax(1))) if labels is not None else None,
              "winning_fit_capped": row.get("hit_cap", False), "winning_fit_near_degenerate": row.get("near_degenerate", False)}
    if test is not None and len(test):
        b = fitted_scores(test, row)
        totals = logsumexp(b, axis=1)
        valid_rows = np.isfinite(totals) & ~np.isnan(b).any(axis=1) & ~np.isposinf(b).any(axis=1)
        prediction_valid = bool(valid_rows.all())
        qt = probabilities(b) if prediction_valid else None
        result.update({"predictive_status": "finite" if prediction_valid else "numerical_failure",
                       "nonfinite_test_rows": int((~valid_rows).sum()),
                       "mean_test_log_density": float(totals.mean()) if prediction_valid else None,
                       "posterior_tv_test": aligned_error(qt, qtrue_test) if qt is not None and qtrue_test is not None else None,
                       "ari_test": float(adjusted_rand_score(test_labels, qt.argmax(1))) if qt is not None and test_labels is not None else None})
    return result


def run_comparison(x, task, output_dir, test=None, labels=None, test_labels=None, qtrue=None, qtrue_test=None, provenance_id=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    config = {**DEFAULTS, **task.get("config", {}), "audit": task.get("audit", False)}
    library = task.get("library", LIBRARIES[task.get("library_id", "L3")])
    k, seed = int(task["fit_k"]), int(task["seed"])
    if provenance_id is None:
        from provenance import fingerprint
        provenance_id, _ = fingerprint()
    signature = digest({"task": task, "config": config, "data": hashlib.sha256(x.tobytes()).hexdigest(),
                        "source_data_runtime_provenance": provenance_id})
    cache = Cache(output_dir / "fits.sqlite", signature)
    started = time.perf_counter()
    try:
        pilot = cache.call("pilot", pilot_fit, (x,k,seed,config), config["fit_timeout"])
        if not pilot.get("finite"):
            raise RuntimeError("Gaussian pilot failed: " + str(pilot.get("error")))
        q = probabilities(fitted_scores(x, pilot))
        table = []
        for j in range(k):
            table.append([cache.call(f"profile:{j}:{name}", profile_cell, (x,q[:,j],name,config), config["profile_timeout"])
                          for name in library])
        if not all(cell.get("finite") for row in table for cell in row):
            write_json(output_dir / "profile_failure.json", table)
            raise RuntimeError("At least one weighted profile failed; screening comparison is unavailable")
        scores = np.array([[cell["score"] for cell in row] for row in table])
        lists, ordered, deficits = shortlists(scores, library, len(x))
        frozen = {"scores": scores, "deficits": deficits, "canonical_shortlists": lists,
                  "ordered_shortlists": ordered, "primary": "joint_2logn",
                  "data_only": True, "stage": "before any non-pilot full-mixture fit",
                  "signature": signature, "thresholds_certified": False}
        existing = cache.get("frozen_screen")
        if existing is not None and digest(existing) != digest(frozen):
            raise RuntimeError("Frozen screen changed during resume")
        cache.put("frozen_screen", frozen)
        write_json(output_dir / "screen.json", frozen)
        common = pilot["seconds"] + sum(cell["seconds"] for row in table for cell in row)
        normal = ("mvn",)*k
        structure_rows, run_rows, polish_rows = {}, {normal: pilot}, {}
        polish_config = {**config, "max_iter": config["polish_max_iter"], "tol": config["polish_tol"]}

        def fit_structure(structure):
            if structure in structure_rows:
                return
            permutations = sorted(set(itertools.permutations(structure)))
            for assignment in permutations:
                if assignment in run_rows:
                    continue
                alpha = [table[j][library.index(name)]["alpha"] for j,name in enumerate(assignment)]
                row = cache.call("fit:" + "+".join(assignment), fit_mixture,
                                 (x, assignment, q.mean(0), alpha, config), config["fit_timeout"])
                row.setdefault("assignment", list(assignment))
                row.setdefault("structure", sorted(assignment))
                run_rows[assignment] = row
            valid = [run_rows[a] for a in permutations if run_rows[a].get("finite")]
            if not valid:
                structure_rows[structure] = None
                return
            best = min(valid, key=lambda row: row["bic"])
            continuation = cache.call("polish:" + "+".join(structure), fit_mixture,
                                      (x, best["assignment"], best["pi"], best["alpha"], polish_config), config["fit_timeout"])
            polish_rows[structure] = continuation
            if continuation.get("finite") and continuation["bic"] < best["bic"]:
                best = continuation
            structure_rows[structure] = best

        # Complete the primary pipeline first, including its numerical refinement.
        for structure in sorted(set(lists["joint_2logn"]) | {normal}):
            fit_structure(structure)
        primary_direct = time.perf_counter()-started if cache.hits == 0 else None
        all_structures = list(itertools.combinations_with_replacement(sorted(library), k))
        for structure in all_structures:
            fit_structure(structure)
        valid = [row for row in structure_rows.values() if row is not None]
        exhaustive_best = min(valid, key=lambda row: row["bic"])
        exhaustive = metrics(exhaustive_best,x,test,labels,test_labels,qtrue,qtrue_test)
        exhaustive.update({"structures": len(all_structures), "assignment_runs": len(run_rows),
                           "continuation_runs": len(polish_rows), "full_fitting_runs": len(run_rows)+len(polish_rows),
                           "accounted_seconds": common+sum(r["seconds"] for a,r in run_rows.items() if a != normal)+sum(r["seconds"] for r in polish_rows.values()),
                           "one_component_profile_full_runs": len(library) if k == 1 else 0,
                           "failed_assignment_runs": sum(not r.get("finite") for r in run_rows.values()),
                           "failed_structures": sum(r is None for r in structure_rows.values()),
                           "failed_continuations": sum(not r.get("finite") for r in polish_rows.values()),
                           "comparison_complete": len(valid)==len(all_structures) and all(r.get("finite") and r.get("monotone",False) for r in run_rows.values()) and all(r.get("finite") and r.get("monotone",False) for r in polish_rows.values()),
                           "capped_assignment_runs": sum(r.get("hit_cap",False) for r in run_rows.values()),
                           "capped_continuations": sum(r.get("hit_cap",False) for r in polish_rows.values()),
                           "nonmonotone_runs": sum(not r.get("monotone",True) for r in [*run_rows.values(),*polish_rows.values()])})
        methods = {}
        for rule, shortlist in lists.items():
            retained = set(shortlist) | {normal}
            feasible = [structure_rows[s] for s in retained if structure_rows[s] is not None]
            winner = min(feasible, key=lambda row: row["bic"])
            assignments = set().union(*(set(itertools.permutations(s)) for s in retained))
            method = metrics(winner,x,test,labels,test_labels,qtrue,qtrue_test)
            method.update({"same_exhaustive_structure": winner["structure"] == exhaustive_best["structure"],
                           "bic_loss": winner["bic"]-exhaustive_best["bic"],
                           "structures_including_pilot": len(library) if k == 1 else len(retained), "assignment_runs": len(assignments),
                           "continuation_runs": sum(s in polish_rows for s in retained),
                           "full_fitting_runs": len(assignments)+sum(s in polish_rows for s in retained)+(len(library) if k == 1 else 0),
                           "one_component_profile_full_runs": len(library) if k == 1 else 0,
                           "accounted_seconds": common+sum(run_rows[a]["seconds"] for a in assignments if a != normal)+sum(polish_rows[s]["seconds"] for s in retained if s in polish_rows),
                           "failed_assignment_runs": sum(not run_rows[a].get("finite") for a in assignments)})
            method["failed_continuations"] = sum(not polish_rows[s].get("finite") for s in retained if s in polish_rows)
            method["agreement_eligible"] = exhaustive["comparison_complete"]
            methods[rule] = method
        # These are diagnostics against a numerical fitted reference, not q-star certificates.
        a_pilot, a_reference = fitted_scores(x,pilot), fitted_scores(x,exhaustive_best)
        rows, cols = linear_sum_assignment(np.array([[np.abs(q[:,j]-probabilities(a_reference)[:,h]).mean() for h in range(k)] for j in range(k)]))
        aligned_reference = np.empty_like(a_reference)
        aligned_reference[:,rows] = a_reference[:,cols]
        logq = a_pilot-logsumexp(a_pilot,axis=1,keepdims=True)
        logr = aligned_reference-logsumexp(aligned_reference,axis=1,keepdims=True)
        b_numeric = float(2*np.sum(q*(logq-logr)))
        penalty = 2*exhaustive_best["loglik"]+exhaustive_best["bic"]
        variational = float(2*np.sum(q*(aligned_reference-logq))-penalty)
        identity_residual = -exhaustive_best["bic"]-variational-b_numeric
        h = aligned_reference-a_pilot
        osc_bound = float(.25*np.sum(np.ptp(h,axis=1)**2))
        deficit_oracle = min(sum(deficits[j,library.index(name)] for j,name in enumerate(a)) for a in set(itertools.permutations(exhaustive_best["structure"])))
        diagnostics = {"twice_kl_to_numerical_reference": b_numeric, "log_score_oscillation_bound": osc_bound,
                       "variational_identity_residual": float(identity_residual),
                       "variational_identity_holds": abs(identity_residual)<=1e-6,
                       "oscillation_inequality_holds": b_numeric <= osc_bound+1e-7,
                       "best_assignment_deficit_of_numerical_winner": float(deficit_oracle),
                       "gap_to_second_numerical_structure": sorted(r["bic"] for r in valid)[1]-exhaustive_best["bic"] if len(valid)>1 else None,
                       "scope": "The reference is a local fitted mixture. Its KL is not a certified B(q) to a global optimum.",
                       "pilot_tv_test": aligned_error(probabilities(fitted_scores(test,pilot)),qtrue_test) if test is not None and qtrue_test is not None else None}
        audit = None
        if config["audit"]:
            # Independent geometric initialization; equal extra work for every structure.
            second = cache.call("audit:pilot", pilot_fit,(x,k,(seed+100003)%2**32,config),config["fit_timeout"])
            if second.get("finite"):
                q2 = probabilities(fitted_scores(x,second))
                table2 = [[cache.call(f"audit:profile:{j}:{name}", profile_cell,(x,q2[:,j],name,config),config["profile_timeout"]) for name in library] for j in range(k)]
                if all(cell.get("finite") for row in table2 for cell in row):
                    improved = dict(structure_rows)
                    audit_runs = []
                    for structure in all_structures:
                        for assignment in sorted(set(itertools.permutations(structure))):
                            alpha = [table2[j][library.index(name)]["alpha"] for j,name in enumerate(assignment)]
                            row = cache.call("audit:fit:"+"+".join(assignment),fit_mixture,(x,assignment,q2.mean(0),alpha,polish_config),config["fit_timeout"])
                            audit_runs.append(row)
                            if row.get("finite") and (improved[structure] is None or row["bic"] < improved[structure]["bic"]):
                                improved[structure] = row
                    audit_best = min((r for r in improved.values() if r is not None),key=lambda row:row["bic"])
                    audit = {"status":"complete", "winner": audit_best["structure"], "bic":audit_best["bic"],
                             "extra_full_runs":1+len(audit_runs), "extra_profiles":k*len(library),
                             "failed_runs":sum(not r.get("finite") for r in audit_runs),
                             "capped_runs":sum(r.get("hit_cap",False) for r in audit_runs),
                             "methods":{rule:{"winner_retained":tuple(audit_best["structure"]) in set(values)|{normal},
                                                "bic_loss":min(improved[s]["bic"] for s in set(values)|{normal} if improved[s] is not None)-audit_best["bic"]} for rule,values in lists.items()},
                             "scope":"Extra validation only; frozen screens unchanged; not counted as algorithm cost."}
            if audit is None:
                audit = {"status":"unavailable", "reason":"Independent pilot or profiles failed"}
        public = None
        if task.get("public_audit", False):
            from public_baseline import public_exhaustive
            public = cache.call("public:exhaustive", public_exhaustive,
                                (x,library,k,{**config,"public_seed":seed}), 3600)
            if public.get("finite"):
                public["primary_retains_public_winner"] = tuple(public["winner"]) in set(lists["joint_2logn"])|{normal}
                public["primary_bic_difference_from_public"] = methods["joint_2logn"]["bic"]-public["bic"]
                public["public_to_primary_direct_time_ratio"] = public["public_workflow_seconds"]/primary_direct if primary_direct else None
        exhaustive["full_fitting_runs"] += len(library) if k == 1 else 0
        return native({"status":"complete", "task":task, "config":config, "signature":signature,
                       "n_train":len(x), "dimension":x.shape[1], "n_test":len(test) if test is not None else 0,
                       "library":library, "exhaustive":exhaustive, "methods":methods,
                       "pilot_seconds":pilot["seconds"], "profile_seconds":common-pilot["seconds"],
                       "profile_count":k*len(library), "profile_numerical_starts":sum(c["numerical_starts"] for r in table for c in r),
                       "primary_direct_seconds":primary_direct, "diagnostics":diagnostics, "audit":audit, "public_audit":public,
                       "elapsed_this_invocation":time.perf_counter()-started, "cache_hits":cache.hits,
                       "truth_used_in_screening":False, "global_optimality_certified":False})
    finally:
        cache.close()
