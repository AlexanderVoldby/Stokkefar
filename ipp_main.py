from __future__ import annotations

# ‑‑‑ stdlib ‑‑‑
import argparse, dataclasses, math, random
from typing import Tuple, Callable
import collections

# ‑‑‑ third‑party ‑‑‑
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd 

# ‑‑‑ local project ‑‑‑
from ipp_queue_sim import (
    Parameters,
    IPPArrivalProcess,
    IPPQueueSimulator,
    BusyPeriodMetric,
    BurstinessMetric,
    SimulationResults,
    EventType,
)

import os, re
def _slug(x: float | int) -> str:
    """Turn 0.25 → '0p25', 5 → '5' so the string is filesystem-safe."""
    return re.sub(r"[\.]", "p", f"{x}")

_FIG_DIR = "figures"
os.makedirs(_FIG_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 0.  IPP ↔ H₂ parameter translation 
# ════════════════════════════════════════════════════════════════════════

def ipp_to_hyper(params: Parameters) -> Tuple[float, float, float]:
    """Return *(p, γ₁, γ₂)* for the H₂ that matches the IPP’s first 2 moments.
    Replace the body once you derive the closed‑form mapping.
    """
    w1 = params.omega_off_to_on
    w2 = params.omega_on_to_off
    lam = params.lambda_on

    p_numerator = (lam - w1 - w2) + math.sqrt((lam + w1 + w2)**2 - 4 * lam * w2)
    p_denominator = math.sqrt((lam + w1 + w2)**2 - 4 * lam * w2)
    p = 1/2 * p_numerator / p_denominator

    gam1 = 1/2 * ((lam + w1 + w2) + p_denominator)
    gam2 = 1/2 * ((lam + w1 + w2) - p_denominator)

    return p, gam1, gam2


def hyper_to_ipp(p: float, g1: float, g2: float) -> Tuple[float, float, float]:
    """Inverse mapping (not required for this project)."""
    lam = p * g1 + (1 - p) * g2
    denominator = p * g1 + (1-p) * g2
    w1 = p * (1-p) * (g1 - g2)**2 / denominator
    w2 = g1 * g2 / denominator

    return w1, w2, lam
# ════════════════════════════════════════════════════════════════════════
# 1.  Utility – sample *n* inter‑arrival times from one IPP
# ════════════════════════════════════════════════════════════════════════

def sample_interarrivals(params: Parameters, n: int = 100_000) -> np.ndarray:
    class _Wrap(random.Random):
        def __init__(self, rng: np.random.Generator):
            self._rng = rng
        def random(self):
            return float(self._rng.random())
        def expovariate(self, rate: float):
            return float(self._rng.exponential(1.0 / rate))

    ipp = IPPArrivalProcess(params, rng=_Wrap(np.random.default_rng(params.seed)))
    out = np.empty(n)
    t = 0.0
    for i in range(n):
        t_next = ipp.next_arrival(t)
        out[i] = t_next - t
        t = t_next
    return out

# ════════════════════════════════════════════════════════════════════════
# 2.  Burstiness diagnostics (CV‑test, KS vs Exp, hazard, ACF)
# ════════════════════════════════════════════════════════════════════════

def run_interarrival_diagnostics(params: Parameters, *, n: int = 100_000):
    samples = sample_interarrivals(params, n=n)
    mean_T, var_T = samples.mean(), samples.var(ddof=1)
    cv2 = var_T / mean_T**2
    Z = (cv2 - 1) * math.sqrt(n) / math.sqrt(2)
    p_cv = 2 * (1 - stats.norm.cdf(abs(Z)))
    D_ks, p_ks = stats.kstest(samples, 'expon', args=(0, mean_T))
    print("\n—— Basic burstiness diagnostics ——")
    print(f" mean inter‑arrival : {mean_T:.4f} s")
    print(f" variance           : {var_T:.4f} s²")
    print(f" C_v²               : {cv2:.4f}")
    print(f" CV‑test            : Z = {Z:.1f}, p = {p_cv:.1e}")
    print(f" KS vs Exp          : D = {D_ks:.4f}, p = {p_ks:.1e}")

    # hazard plot
    sorted_s = np.sort(samples)
    surv = 1.0 - np.arange(1, n + 1) / n
    surv = np.maximum(surv, 1e-12)
    knots = sorted_s[::1000]
    hazard = -np.gradient(np.log(surv[::1000]), knots)
    plt.figure(); plt.plot(knots, hazard)
    plt.xlabel('t (s)'); plt.ylabel('h(t)'); plt.title('Hazard rate'); plt.tight_layout(); plt.savefig('hazard.png', dpi=300)

    # acf plot (30 lags)
    def _acf(x, m=30):
        x = x - x.mean(); var = (x**2).mean()
        return [1.0] + [np.dot(x[:-k], x[k:]) / ((len(x)-k)*var) for k in range(1, m+1)]
    acf_vals = _acf(samples)
    plt.figure(); plt.stem(range(len(acf_vals)), acf_vals, basefmt=' ')
    plt.xlabel('Lag'); plt.ylabel('ACF'); plt.title('Inter‑arrival ACF'); plt.tight_layout(); plt.savefig('acf.png', dpi=300)
    print(' hazard.png and acf.png written.')

# ════════════════════════════════════════════════════════════════════════
# 3.  H₂ goodness‑of‑fit diagnostics
# ════════════════════════════════════════════════════════════════════════

def _cdf_h2(x, p, g1, g2):
    x = np.asarray(x)
    return 1.0 - p * np.exp(-g1 * x) - (1 - p) * np.exp(-g2 * x)

def _pdf_h2(x, p, g1, g2):
    x = np.asarray(x)
    return p * g1 * np.exp(-g1 * x) + (1 - p) * g2 * np.exp(-g2 * x)

def run_h2_test(params: Parameters, *, n: int = 100_000,
                bins: int = 80, tail_q: float = 0.995):
    """
    KS test + histogram / pdf overlay for the H₂ approximation.

    tail_q : keep only the lower *tail_q* fraction of the sample when plotting.
             The KS test still uses the full data set.
    """
    samples = sample_interarrivals(params, n=n)

    # full-sample KS test (unchanged)
    p, g1, g2 = ipp_to_hyper(params)
    D, p_val  = stats.kstest(samples, lambda x: _cdf_h2(x, p, g1, g2))
    print("\n—— H₂ goodness-of-fit ——")
    print(f" p, γ1, γ2 : {p:.4f}, {g1:.4f}, {g2:.4f}")
    print(f" KS stat   : {D:.4f}, p = {p_val:.2e}")

    # ----------  plotting domain  ------------------------------------
    x_max = np.quantile(samples, tail_q)   # e.g. 99.5-percentile
    xs    = np.linspace(0, x_max, 400)

    plt.figure()
    plt.hist(samples[samples <= x_max], bins=bins, range=(0, x_max),
             density=True, alpha=0.6, label="sample")
    plt.plot(xs, _pdf_h2(xs, p, g1, g2), "k-", lw=2, label="H₂ pdf")
    plt.xlabel("t  (s)")
    plt.ylabel("density")
    plt.title(f"Histogram truncated at {tail_q*100:.1f}-percentile (x ≤ {x_max:.1f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("h2_hist.png", dpi=300)
    print(" h2_hist.png written.")
# ════════════════════════════════════════════════════════════════════════
# 4.  Queue‑simulation helpers
# ════════════════════════════════════════════════════════════════════════

def run_single_ipp(args):
    p = Parameters(omega_off_to_on=args.omega1, omega_on_to_off=args.omega2,
                   lambda_on=args.lam_on, mu=args.mu, horizon=args.horizon,
                   seed=args.seed)
    sim = IPPQueueSimulator(p, metrics=[BusyPeriodMetric(), BurstinessMetric()])
    res = sim.run(); res.pretty_print(p)
    if args.verify_mm1:
        p_mm1 = dataclasses.replace(p, omega_off_to_on=1e6, omega_on_to_off=1e-6)
        theo = p_mm1.lambda_on / (p_mm1.mu * (1 - p_mm1.lambda_on / p_mm1.mu))
        wq_sim = IPPQueueSimulator(p_mm1).run().wait_times.mean()
        print("\n—— M/M/1 verification ——\n Simulated W_q : {:.4f}\n Analytic  W_q : {:.4f}".format(wq_sim, theo))
    if args.diag:
        run_interarrival_diagnostics(p, n=args.diag_samples)
    if args.h2_test:
        run_h2_test(p, n=args.diag_samples)


def _simulate_superposed(args, base: Parameters, k: int, *, bins: int,
                         log_events: bool = True):
    """
    Helper returning (wait_times, queue_len_samples, stats_dict) for one *k*.
    """
    sim = IPPQueueSimulator(
        base,
        metrics=[BusyPeriodMetric()],   # busy-period data now collected
        log_events=log_events
    )

    # --- replace single stream by superposition of *k* streams ----------
    if base.seed is None:
        # fully random run (time-based seed inside Random())
        rngs = [random.Random() for _ in range(k)]
    else:
        master = random.Random(base.seed)           # seeded once
        rngs = [random.Random(master.getrandbits(32)) for _ in range(k)]
    # Now build the IPP arrival streams
    ipps = [IPPArrivalProcess(base, rng=r) for r in rngs]

    class _Super:
        def __init__(self, streams):
            self._s = streams
            self._next = [ip.next_arrival(0.0) for ip in streams]

        def next_arrival(self, now: float):
            i = int(np.argmin(self._next))
            t = self._next[i]
            self._next[i] = self._s[i].next_arrival(t)
            return t

    sim.arrivals = _Super(ipps)
    sim.next_arrival = sim.arrivals.next_arrival(sim.t)

    # --------------------- run & harvest --------------------------------
    res = sim.run()
    wq = res.wait_times
    qs = np.asarray([ev.queue_len for ev in res.event_log])

    # NEW — theoretical ρ for this k
    lambda_eff_one = base.pi_on * base.lambda_on       # long-run λ of one IPP
    rho_theo       = k * lambda_eff_one / base.mu
    rho_sim        = res.utilisation                  # busy_time / horizon

    def _ci95(x):
        if len(x) < 2:
            return np.nan
        return stats.t.ppf(0.975, df=len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


    stats_dict = {
        "k": k,
        "rho_th": rho_theo,
        "rho_sim": rho_sim,
        "E[Wq]": wq.mean(),
        "Var[Wq]": wq.var(ddof=1),
        "95%-CI Wq": _ci95(wq),
        "E[Lq]": qs.mean(),
        "Var[Lq]": qs.var(ddof=1),
        "95%-CI Lq": _ci95(qs),
        "Lq_time_avg": res.metrics.get("L_time_avg", np.nan),
    }

    if args.dep_diag:
        run_departure_and_busy_diagnostics(res, base, k=k, bins=args.bins)

    return wq, qs, stats_dict

# ════════════════════════════════════════════════════════════════════════

def run_multi_ipp(args):
    """
    Superpose *k* identical IPPs.

    For every k supplied with ``-k`` it now dumps two separate files
       * wait_times_k<k>.png   (histogram of W_q)
       * queue_len_k<k>.png    (PMF of L_q)
    and prints the per-k summary table just like before.
    """
    ks: list[int] = args.k
    base = Parameters(
        omega_off_to_on=args.omega1,
        omega_on_to_off=args.omega2,
        lambda_on=args.lam_on,
        mu=args.mu,
        horizon=args.horizon,
        seed=args.seed,
    )

    all_stats = []
    for k in ks:
        wq, qs, st = _simulate_superposed(args, base, k, bins=args.bins)
        all_stats.append(st)

        prefix = (
            f"k{_slug(k)}"
            f"_mu{_slug(base.mu)}"
            f"_w1{_slug(base.omega_off_to_on)}"
            f"_w2{_slug(base.omega_on_to_off)}"
            f"_lam{_slug(base.lambda_on)}"
        )

        fname_wq = f"{_FIG_DIR}/wait_times_{prefix}.png"
        fname_q  = f"{_FIG_DIR}/queue_len_{prefix}.png"

        # ——— waiting-time histogram ——————————————
        fig_wq, ax_wq = plt.subplots()
        ax_wq.hist(wq, bins=args.bins, density=True, alpha=0.7)
        ax_wq.set_xlabel("Waiting time  Wq  (s)")
        ax_wq.set_ylabel("density")
        ax_wq.set_title(
            f"Wq dist  (k={k}, μ={base.mu}, ω₁={base.omega_off_to_on}, "
            f"ω₂={base.omega_on_to_off}, λON={base.lambda_on})"
        )
        fig_wq.tight_layout()
        fig_wq.savefig(fname_wq, dpi=300)
        plt.close(fig_wq)                         # free memory

        # ——— queue-length PMF ————————————————
        vals, cnts = np.unique(qs, return_counts=True)
        pmf = cnts / cnts.sum()
        fig_q, ax_q = plt.subplots()
        ax_q.stem(vals, pmf, basefmt=" ")
        ax_q.set_xlabel("Queue length  Lq")
        ax_q.set_ylabel("probability")
        ax_q.set_title(
            f"Lq PMF   (k={k}, μ={base.mu}, ω₁={base.omega_off_to_on}, "
            f"ω₂={base.omega_on_to_off}, λON={base.lambda_on})"
        )
        fig_q.tight_layout()
        fig_q.savefig(fname_q, dpi=300)
        plt.close(fig_q)

        print(f"  → {fname_wq} and {fname_q} written.")

    # ——— statistics table (unchanged) ——————————
    df = pd.DataFrame(all_stats).set_index("k")
    print("\n———— Summary statistics ————")
    print(df.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("———————————————————————————")

# ════════════════════════════════════════════════════════════════════
# D.  Departure-process & busy-period diagnostics
# ════════════════════════════════════════════════════════════════════
def run_departure_and_busy_diagnostics(results: SimulationResults,
                                       params: Parameters,
                                       *,
                                       k: int = 1,
                                       bins: int = 80):
    # --- extract inter-departure gaps ---------------------------------
    deps   = [ev.time for ev in results.event_log if ev.event is EventType.DEPARTURE]
    deps   = np.asarray(deps, dtype=float)
    idgaps = np.diff(deps)                        # inter-departure gaps

    # ----- basic stats -------------------------------------------------
    mean_D, var_D = idgaps.mean(), idgaps.var(ddof=1)
    cv2_D = var_D / mean_D**2
    Z = (cv2_D - 1) * math.sqrt(len(idgaps)) / math.sqrt(2)
    p_cv  = 2 * (1 - stats.norm.cdf(abs(Z)))
    D_ks, p_ks = stats.kstest(idgaps, 'expon', args=(0, mean_D))

    print("\n—— Departure-stream diagnostics ——")
    print(f" mean inter-dep : {mean_D:.4f} s")
    print(f" variance       : {var_D:.4f} s²")
    print(f" C_v²           : {cv2_D:.4f}")
    print(f" CV-test        : Z = {Z:.1f}, p = {p_cv:.1e}")
    print(f" KS vs Exp      : D = {D_ks:.4f}, p = {p_ks:.1e}")

    # ----- plot histogram & Exp pdf -----------------------------------
    xs = np.linspace(0, np.quantile(idgaps, 0.995), 400)
    plt.figure()
    plt.hist(idgaps, bins=bins, density=True, alpha=0.6, label="sample")
    plt.plot(xs, stats.expon.pdf(xs, scale=mean_D), 'k-', lw=2, label="Exp fit")
    plt.xlabel("t (s)"); plt.ylabel("density"); plt.title("Inter-departure pdf")
    f_dep = f"figures/dep_pdf_k{k}_mu{params.mu}.png"
    plt.tight_layout(); plt.savefig(f_dep, dpi=300)
    print(f" {f_dep} written.")

    # ----- busy-period stats & hist -----------------------------------
    bp_lengths = np.asarray(results.metrics['busy_periods']['lengths'], dtype=float)
    if bp_lengths.size == 0:
        print(" No complete busy period in horizon — likely ρ ≥ 1.")
        return

    mean_B = bp_lengths.mean()
    var_B  = bp_lengths.var(ddof=1)
    lam_eff = k * params.pi_on * params.lambda_on
    theo_rate = params.mu - lam_eff
    plt.figure()
    plt.hist(bp_lengths, bins=bins, density=True, alpha=0.6, label="sample")
    xs = np.linspace(0, np.quantile(bp_lengths, 0.995), 400)
    plt.plot(xs, stats.expon.pdf(xs, scale=1/theo_rate), 'k-', lw=2,
             label=f"Exp(μ−λ_eff) [{theo_rate:.2f}]")
    plt.xlabel("t (s)"); plt.ylabel("density"); plt.title("Busy-period pdf")
    f_bp = f"figures/busy_pdf_k{k}_mu{params.mu}.png"
    plt.tight_layout(); plt.savefig(f_bp, dpi=300)

    print("\n—— Busy-period statistics ——")
    print(f" mean B       : {mean_B:.4f} s   (theory: {1/theo_rate:.4f})")
    print(f" variance     : {var_B:.4f} s²  (theory: {1/theo_rate**2:.4f})")
    print(f" {f_bp} written.")

def run_diag_only(args):
    """
    Stand-alone arrival-stream diagnostics (no queue simulation).

    Always performs the CV-test, KS-vs-Exponential test, hazard-rate
    plot and ACF plot on *args.diag_samples* inter-arrival gaps.

    If the user supplies ``--h2-test`` it additionally runs the
    IPP → H₂ Kolmogorov–Smirnov test and saves a histogram overlay.
    """
    # A minimal Parameters object – μ and horizon are irrelevant here.
    p = Parameters(
        omega_off_to_on=args.omega1,
        omega_on_to_off=args.omega2,
        lambda_on=args.lam_on,
        mu=args.mu,
        horizon=1.0,
        seed=args.seed,
    )

    # Core burstiness diagnostics (always executed)
    run_interarrival_diagnostics(p, n=args.diag_samples)

    # Optional hyper-exponential goodness-of-fit
    if getattr(args, "h2_test", False):
        run_h2_test(p, n=args.diag_samples)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  CLI plumbing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("IPP queue project driver")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--omega1", type=float, default=0.2)
    common.add_argument("--omega2", type=float, default=0.3)
    common.add_argument("--lam-on", type=float, dest="lam_on", default=1.5)
    common.add_argument("--mu", type=float, default=2.0)
    common.add_argument("--horizon", type=float, default=1e5)

    # single ------------------------------------------------------------
    s1 = sub.add_parser("single", parents=[common])
    s1.add_argument("--verify-mm1", action="store_true")
    s1.add_argument("--diag", action="store_true")
    s1.add_argument("--diag-samples", type=int, default=100_000)
    s1.add_argument("--h2-test", action="store_true", help="Run IPP→H2 KS test & histogram")
    s1.set_defaults(func=run_single_ipp)

    # ––– multi –––
    s2 = sub.add_parser(
        "multi",
        parents=[common],
        help="Superpose k identical IPPs (single or sweep)",
    )
    s2.add_argument(
        "-k",
        type=int,
        nargs="+",
        default=[3],             # backwards-compatible default
        help="One or more values of k (e.g. -k 1 2 4 8)",
    )
    s2.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots when multiple k are given "
             "(implied when len(k) > 1)",
    )
    s2.add_argument("--bins", type=int, default=80, help="Histogram bin count")
    s2.add_argument("--dep-diag", action="store_true",
                help="Departure & busy-period diagnostics")
    s2.set_defaults(func=run_multi_ipp)
    

    # ––– diag only –––
    s3 = sub.add_parser("diag", parents=[common], help="Only run inter-arrival diagnostics")
    s3.add_argument("--diag-samples", type=int, default=100_000, help="Sample size")
    s3.add_argument("--h2-test", action="store_true",
                    help="Run IPP→H2 KS test & histogram")
    s3.set_defaults(func=run_diag_only)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
