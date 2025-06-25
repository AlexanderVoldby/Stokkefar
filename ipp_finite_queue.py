from __future__ import annotations
"""
Finite–capacity extension of the IPP/M/1 simulator.

The code re‑uses the existing `IPPQueueSimulator` implementation but adds

*   a hard system capacity **K** (customers in queue **+** at most one in service),
*   blocking / loss when an arrival finds the system full, and
*   statistics for the blocking probability.

Example usage (single run):

```bash
python ipp_finite_queue.py --seed 42 --capacity 10 --omega1 0.2 --omega2 0.3 \
                           --lam-on 1.5 --mu 2.0 --horizon 1e5
```

You can also superpose several IPPs with the `-k` flag exactly like in
`ipp_main.py`:

```bash
python ipp_finite_queue.py multi -k 1 2 4 8 --capacity 20 --mu 2.5
```

The script prints blocking probability, mean waiting time of **accepted**
customers, etc., and optionally saves the same plots as the infinite‑capacity
version.
"""

# ──────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────
import argparse
import math
import random
from pathlib import Path
from typing import Sequence, List

# ──────────────────────────────────────────────────────────────────────
# Third‑party
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────────────────────────────
# Local project (original infinite‑capacity engine)
# ──────────────────────────────────────────────────────────────────────
from ipp_queue_sim import (
    Parameters,
    IPPArrivalProcess,
    IPPQueueSimulator,
    BusyPeriodMetric,
    EventType,
)

# ──────────────────────────────────────────────────────────────────────
# 1.  Finite‑capacity queue simulator
# ──────────────────────────────────────────────────────────────────────
class FiniteCapacityIPPQueueSimulator(IPPQueueSimulator):
    """IPP/M/1/K queue – lost customers are counted, not queued."""

    def __init__(self, params: Parameters, *, capacity: int,
                metrics=None, log_events=False):
        self.K = int(capacity)
        super().__init__(params, metrics=metrics, log_events=log_events)
        self.blocked = 0  # lost customers
        self.arrivals_total: int = 0

    # ------------------------------------------------------------------
    def _handle_arrival(self):  # override to inject capacity check
        # advance area under q(t) if the patch was applied in ipp_queue_sim
        if hasattr(self, "_advance_area"):
            self._advance_area(self.next_arrival, len(self.queue))

        self.t = self.next_arrival
        self.arrivals_total += 1

        # current system size = jobs waiting + (1 if server busy)
        sys_len = len(self.queue) + (1 if self.t < self.server_busy_until else 0)
        if sys_len >= self.K:
            # --- BLOCKED ------------------------------------------------
            self.blocked += 1
            # schedule *next* arrival from the IPP but DO NOT add customer
            self.next_arrival = self.arrivals.next_arrival(self.t)
            # log an arrival event flagged as blocked (optional)
            if self.event_log is not None:
                self.event_log.append(
                    dict(time=self.t, queue_len=sys_len, event="BLOCKED")
                )
            return  # early exit – nothing else changes

        # --- ACCEPTED customer – fall back to original logic -------------
        svc = self.rng.expovariate(self.params.mu)
        if self.t >= self.server_busy_until:  # immediate service
            self.server_busy_until = self.t + svc
            self.next_departure = self.server_busy_until
            self.busy_time += svc
            self.wait_times.append(0.0)
        else:  # joins queue
            wait = (self.server_busy_until - self.t) + sum(self.queue)
            self.wait_times.append(wait)
            self.queue.append(svc)

        # bookkeeping identical to parent class
        self.service_times.append(svc)
        self.next_arrival = self.arrivals.next_arrival(self.t)
        self._log_event(EventType.ARRIVAL)

    # ------------------------------------------------------------------
    def run(self):  # extend parent to add blocking stats
        res = super().run()
        res.metrics["blocked"] = self.blocked
        res.metrics["P_block"] = self.blocked / self.arrivals_total if self.arrivals_total else math.nan
        res.metrics["arrivals_total"] = self.arrivals_total
        return res

# ──────────────────────────────────────────────────────────────────────
# 2.  Helper – one experiment with given k and capacity
# ──────────────────────────────────────────────────────────────────────

def simulate_k(base: Parameters, *, k: int, K: int, seed: int | None, bins: int = 50):
    # master RNG for sub‑seeds (reproducible independence)
    master = random.Random(seed)
    rngs = [random.Random(master.getrandbits(32)) for _ in range(k)]
    ipps = [IPPArrivalProcess(base, rng=r) for r in rngs]

    # wrap them just like in the infinite‑capacity code
    class _Super:
        def __init__(self, streams):
            self._s = streams
            self._next = [ip.next_arrival(0.0) for ip in streams]
        def next_arrival(self, now):
            i = int(np.argmin(self._next))
            t = self._next[i]
            self._next[i] = self._s[i].next_arrival(t)
            return t

    sim = FiniteCapacityIPPQueueSimulator(
          base, capacity=K,
          metrics=[BusyPeriodMetric()], log_events=True)
    
    sim.arrivals = _Super(ipps)
    sim.next_arrival = sim.arrivals.next_arrival(sim.t)

    res = sim.run()
    blk_prob = res.metrics["P_block"]

    # quick console summary (extend or format as needed)
    print(f"k={k:2d}  blocked={blk_prob:6.3%}  mean Wq={res.wait_times.mean():.2f}  time‑avg Lq={res.metrics['L_time_avg']:.2f}")
    return res

# ──────────────────────────────────────────────────────────────────────
# New CLI that supports *lists* for every numeric parameter
# ──────────────────────────────────────────────────────────────────────
import itertools, argparse
from typing import List

def _float_list(xs: List[str]) -> List[float]:
    return [float(x) for x in xs]

def _int_list(xs: List[str]) -> List[int]:
    return [int(x) for x in xs]

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "IPP/M/1/K simulator – finite capacity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=None,
                    help="base RNG seed (one master seed for all runs)")

    # ---- global option: capacity ------------------------------------------------
    ap.add_argument("-K", "--capacity", metavar="CAP", type=_int_list, nargs="+",
                    required=True, help="system capacity K (one or more)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- common arrival/service parameters --------------------------------------
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--omega1", type=_float_list, nargs="+", default=[0.2])
    common.add_argument("--omega2", type=_float_list, nargs="+", default=[0.3])
    common.add_argument("--lam-on", type=_float_list, nargs="+",
                        dest="lam_on", default=[1.5])
    common.add_argument("--mu", type=_float_list, nargs="+", default=[2.0])
    common.add_argument("--horizon", type=float, default=1e5)
    common.add_argument("--bins", type=int, default=80)

    # ---- single-k convenience ----------------------------------------------------
    s1 = sub.add_parser("single", parents=[common])
    s1.add_argument("-k", type=_int_list, nargs="+", default=[3])
    s1.set_defaults(func=run_grid)           # will execute the grid anyway

    # ---- multi-k sweep -----------------------------------------------------------
    s2 = sub.add_parser("multi", parents=[common])
    s2.add_argument("-k", type=_int_list, nargs="+", default=[1, 2, 3, 4])
    s2.set_defaults(func=run_grid)

    return ap


# ──────────────────────────────────────────────────────────────────────
# Grid runner (replaces run_single / run_multi dispatch)
# ──────────────────────────────────────────────────────────────────────
def run_grid(args):
    """
    Iterate over the Cartesian product of all lists supplied on the CLI,
    run the finite-capacity simulator once per combination, and collect
    the results in a single pandas DataFrame.
    """
    param_space = itertools.product(
        args.capacity,
        args.k,
        args.omega1,
        args.omega2,
        args.lam_on,
        args.mu,
    )

    all_stats = []
    idx = 0
    for (K, k, w1, w2, lam_on, mu) in param_space:
        idx += 1
        print(f"\n### Run {idx}: K={K}  k={k}  ω1={w1}  ω2={w2}  "
              f"λON={lam_on}  μ={mu}")
        base = Parameters(
            omega_off_to_on=w1,
            omega_on_to_off=w2,
            lambda_on=lam_on,
            mu=mu,
            horizon=args.horizon,
            seed=args.seed,
        )
        # single helper that already exists in the file
        wq, qs, st = simulate_k(
            base, k=k, K=K, seed=args.seed,
            bins=args.bins)
        all_stats.append({
            **st,
            "K": K,
            "omega1": w1,
            "omega2": w2,
            "lam_on": lam_on,
            "mu": mu,
        })

    # ---------- summary -------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(all_stats)
    cols = ["K", "k", "omega1", "omega2", "lam_on", "mu",
            "rho_th", "rho_sim",
            "E[Wq]", "E[Lq]", "Lq_time_avg", "p_block"]
    print("\n———— Full grid summary ————")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:10.4f}"))
    print("———————————————————————————")

def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
