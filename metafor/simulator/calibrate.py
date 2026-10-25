"""
Calibrate a DAG's service rates so that the *simulated* saturation throughput
matches a target peak workload.

Why this is needed: the path capacity of a DAG is not simply
``threads x service_rate``. In the simulator a thread stays busy while it waits
for its downstream call, so capacity is governed by ``threads`` divided by the
sum of service times along the path (and by fan-out/visit ratios). Per-server
trace data therefore cannot fix the absolute scale of the service rates.
Instead we keep their *relative* values (taken from exclusive handler times)
and apply one global factor ``k`` so that

    saturation goodput of the simulated system == target_goodput * peak_rps.

``target_goodput`` is the fraction of the peak workload the system is meant to
complete when offered ``peak_rps``. It is 1.0 when the system just keeps up with
the peak, and below 1.0 when the traced peak already overloads it (a system
offered ``peak_rps`` that is overloaded completes only about its capacity).

Capacity scales linearly with ``k`` (exponential service, negligible network
delay), so the factor converges in one or two corrections.
"""
import contextlib
import io
import logging
import random
import sys
from dataclasses import replace

from metafor.simulator.dag import DAG
from metafor.simulator.job import ExponentialDistribution


def scale_dag(dag: DAG, k: float) -> DAG:
    """Return a copy of `dag` whose exponential service rates are multiplied by k."""
    nodes = []
    for n in dag:
        sd = n.service_dist
        if not isinstance(sd, ExponentialDistribution):
            raise TypeError(
                f"node {n.node_id}: calibration only supports ExponentialDistribution "
                f"service times, got {type(sd).__name__}"
            )
        nodes.append(replace(n, service_dist=ExponentialDistribution(sd.rate * k)))
    return DAG(nodes)


def scale_population(dag: DAG, factor: float, verbose: bool = True) -> DAG:
    """
    Return a copy of `dag` that carries `factor` times fewer requests: thread
    pools and queue sizes are divided by `factor`; per-thread service rates and
    all timeouts are left unchanged. The caller must divide the offered load
    (peak_rps) by the same factor.

    Dividing arrivals, servers and buffers together keeps utilization, the
    time a queued request waits, and the timeout-to-queueing-delay ratio the
    same, which is what overload and retry dynamics depend on, while cutting the
    number of simulated events by about `factor`. It does not preserve
    queueing at moderate load: with fewer servers per node, requests wait more
    often than they would with a large pool (fewer servers, more waiting).

    Thread and queue counts are integers >= 1. When threads / factor is not a
    whole number, or drops below 1, the node's capacity is kept at
    capacity / factor by adjusting its per-thread rate, which changes that
    node's service time; this is reported when `verbose`.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")

    nodes = []
    for n in dag:
        sd = n.service_dist
        if not isinstance(sd, ExponentialDistribution):
            raise TypeError(
                f"node {n.node_id}: scaling only supports ExponentialDistribution "
                f"service times, got {type(sd).__name__}"
            )
        ideal_threads = n.threads / factor
        threads = max(1, round(ideal_threads))
        queue = max(1, round(n.queue_size / factor))
        rate = sd.rate * ideal_threads / threads   # keep node capacity = capacity / factor

        if verbose and (abs(rate / sd.rate - 1.0) > 0.10 or queue * factor < 0.5 * n.queue_size
                        or queue * factor > 1.5 * n.queue_size):
            print(f"[scale] node {n.node_id}: threads {n.threads} -> {threads} "
                  f"(ideal {ideal_threads:.2f}), queue {n.queue_size} -> {queue}; "
                  f"per-thread service time x{sd.rate / rate:.2f}",
                  file=sys.stderr)
        nodes.append(replace(n, threads=threads, queue_size=queue,
                             service_dist=ExponentialDistribution(rate)))
    return DAG(nodes)


def measure_goodput(dag: DAG, offered_rps: float, sim_time: float = 200.0,
                    warmup_frac: float = 0.3, seed: int = 0) -> float:
    """
    Simulate `dag` under a constant open-loop load of `offered_rps` requests/s
    per entry node, with timeouts and retries disabled so that every completion
    counts, and return the steady-state completion rate per entry node.
    """
    # Imported here to avoid a circular import (simulate imports dag, client, ...).
    from metafor.simulator.simulate import make_sim_exp, Simulator

    # No server-side timeouts/retries. Client timeout is effectively infinite;
    # max_retries must stay >= 1 because the server drops queued jobs older
    # than client.max_retries * client.timeout.
    nodes = [replace(n, timeout=None, max_retries=0) for n in dag]
    cdag = DAG(nodes)

    prev_disable = logging.root.manager.disable
    rng_state = random.getstate()
    logging.disable(logging.CRITICAL)
    try:
        random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            servers, clients = make_sim_exp(
                1.0 / offered_rps, "calibration", "request",
                1.0, 0, 1e12, 1,             # rho, queue_size (unused), timeout_t, max_retries
                1.0, 1.0,                     # rho_fault, rho_reset (no fault)
                [sim_time * 10, sim_time * 20], 0.0,
                False, 0.9, 0.5, "fifo", "exp", cdag, 0,
            )
            sim = Simulator(servers, clients, cdag, 0)
            sim.reset()
            warm = sim_time * warmup_frac
            sim.sim(warm)
            n0 = sum(c.num_complete_jobs for c in clients)
            sim.sim(sim_time)
            n1 = sum(c.num_complete_jobs for c in clients)
    finally:
        logging.disable(prev_disable)
        random.setstate(rng_state)

    return (n1 - n0) / (sim_time - warm) / len(clients)


def calibrate_dag(dag: DAG, peak_rps: float, target_goodput: float = 1.0,
                  overload: float = 2.0, sim_time: float = 200.0, tol: float = 0.03,
                  max_iter: int = 8, max_offered_factor: float = 64.0, seed: int = 0,
                  verbose: bool = False):
    """
    Find k such that scale_dag(dag, k) saturates at target_goodput * peak_rps,
    i.e. completes `target_goodput` (a fraction in (0, 1]) of the peak workload
    when offered `peak_rps`.

    Returns (calibrated_dag, k). Each iteration drives the system at
    `overload * peak_rps`; if the system is not saturated at that load (it
    completes ~everything offered) its capacity is above the offered load, so
    the load is raised instead of rescaling, because only a saturated
    measurement reveals capacity.
    """
    if peak_rps <= 0:
        raise ValueError("peak_rps must be > 0")
    if not (0.0 < target_goodput <= 1.0):
        raise ValueError("target_goodput must be in (0, 1]")

    target_rps = target_goodput * peak_rps
    k = 1.0
    base_offered = overload * peak_rps
    offered = base_offered
    converged = False

    for it in range(max_iter):
        x = measure_goodput(scale_dag(dag, k), offered, sim_time=sim_time, seed=seed)
        saturated = x < 0.9 * offered
        if verbose:
            print(f"[calibrate] iter {it}: k={k:.4f} offered={offered:.1f} "
                  f"goodput={x:.1f} saturated={saturated}")

        if not saturated:
            if offered >= max_offered_factor * peak_rps:
                print(f"[calibrate] warning: system not saturated even at "
                      f"{offered:.0f} req/s; keeping k={k:.4f}")
                break
            offered *= 4.0
            continue

        if abs(x / target_rps - 1.0) <= tol:
            converged = True
            break

        k *= target_rps / x
        offered = base_offered

    if not converged:
        print(f"[calibrate] warning: did not converge to within {tol:.0%} of "
              f"target goodput {target_rps:g} req/s; using k={k:.4f}")

    return scale_dag(dag, k), k


def calibrate_at_peak(dag: DAG, peak_rps: float, target_goodput: float = 1.0,
                      overload: float = 2.0, sim_time: float = 200.0,
                      tol: float = 0.02, max_refine: int = 8,
                      min_completions: int = 5000, max_sim_time: float = 3600.0,
                      seed: int = 0, verbose: bool = False):
    """
    Calibrate `dag` so that, when offered exactly `peak_rps`, it completes
    `target_goodput * peak_rps` requests per second (within `tol`).

    Two stages. First `calibrate_dag` matches the *saturation* throughput to the
    target, which is cheap and gets close. Then the goodput is measured at the
    real operating point (offered == peak_rps) and k is corrected until that
    measurement hits the target, because near saturation the throughput at the
    peak differs slightly from the plateau (finite queues, queueing delay).

    The measurement window is lengthened if needed so that it holds about
    `min_completions` completions; shorter windows make the measurement noisier
    than `tol`.

    Returns (calibrated_dag, report). `report` is JSON-serialisable and records
    the scale and the achieved goodput so callers can store and reuse it.
    """
    if peak_rps <= 0:
        raise ValueError("peak_rps must be > 0")
    if not (0.0 < target_goodput <= 1.0):
        raise ValueError("target_goodput must be in (0, 1]")

    target_rps = target_goodput * peak_rps
    warmup_frac = 0.3
    sim_eff = min(max_sim_time,
                  max(sim_time, min_completions / ((1.0 - warmup_frac) * target_rps)))

    _, k = calibrate_dag(dag, peak_rps, target_goodput=target_goodput,
                         overload=overload, sim_time=sim_eff, tol=2 * tol,
                         seed=seed, verbose=verbose)
    plateau_k = k

    best = None
    converged = False
    iterations = 0
    for it in range(max_refine + 1):
        iterations = it
        x = measure_goodput(scale_dag(dag, k), peak_rps, sim_time=sim_eff,
                            warmup_frac=warmup_frac, seed=seed)
        err = x / target_rps - 1.0
        if best is None or abs(err) < abs(best[2]):
            best = (k, x, err)
        if verbose:
            print(f"[calibrate] refine {it}: k={k:.4f} goodput at offered={peak_rps:g}: "
                  f"{x:.2f} req/s (target {target_rps:.2f}, error {err:+.1%})")
        if abs(err) <= tol:
            converged = True
            break
        k *= 2.0 if x <= 0 else min(2.0, max(0.5, target_rps / x))

    k, x, err = best
    if not converged:
        print(f"[calibrate] warning: goodput at peak is {x:.2f} req/s, "
              f"{err:+.1%} from the target {target_rps:.2f} req/s "
              f"(tolerance {tol:.1%}); using the closest scale found, k={k:.4f}")

    report = {
        "scale": k,
        "plateau_scale": plateau_k,
        "peak_rps": peak_rps,
        "target_goodput": target_goodput,
        "target_rps": target_rps,
        "measured_rps": x,
        "measured_fraction": x / peak_rps,
        "relative_error": err,
        "tolerance": tol,
        "converged": converged,
        "refine_iterations": iterations,
        "sim_time_used": sim_eff,
    }
    return scale_dag(dag, k), report