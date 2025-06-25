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
        self.attempted = 0  # total arrivals attempted
        self.blocked = 0  # lost customers
        self.arrivals_total: int = 0

    # ------------------------------------------------------------------
    def _handle_arrival(self):        # override for K-limited system
        # 1. advance clock ───────────────────────────────────────────────
        self.t = self.next_arrival

        # optional: keep area-under-q(t) integration in sync
        if hasattr(self, "_advance_area"):
            self._advance_area(self.t, len(self.queue))

        # 2. account the attempt ─────────────────────────────────────────
        self.attempted += 1

        # 3. capacity check ──────────────────────────────────────────────
        if len(self.queue) >= self.K:
            # -- blocked --------------------------------------------------
            self.blocked += 1
            self.next_arrival = self.arrivals.next_arrival(self.t)

            # log the rejected attempt (optional but usually wanted)
            self._log_event(EventType.ARRIVAL)
            return

        # 4. accepted customer – identical to parent implementation ─────
        svc = self.rng.expovariate(self.params.mu)

        if self.t >= self.server_busy_until:  # immediate service
            self.server_busy_until = self.t + svc
            self.next_departure    = self.server_busy_until
            self.busy_time        += svc
            self.wait_times.append(0.0)
        else:                                  # joins queue
            wait = (self.server_busy_until - self.t) + sum(self.queue)
            self.wait_times.append(wait)
            self.queue.append(svc)

        # bookkeeping
        self.service_times.append(svc)
        self.next_arrival = self.arrivals.next_arrival(self.t)
        self._log_event(EventType.ARRIVAL)


    # ------------------------------------------------------------------
    def run(self):
        """
        Same event-loop as the parent class, but afterwards we attach two
        extra attributes (`total_arrivals`, `blocked`) and record the
        blocking probability in the `metrics` dictionary.
        """
        res = super().run()                     # SimulationResults instance

        # -----------------------------------------------------------------
        # add NEW attributes that downstream code can access directly
        # -----------------------------------------------------------------
        res.total_arrivals = self.attempted     # how many arrival attempts
        res.blocked        = self.blocked       # how many were rejected

        # -----------------------------------------------------------------
        # store nicely formatted metrics (so they appear in pretty_print)
        # -----------------------------------------------------------------
        p_block = (self.blocked / self.attempted) if self.attempted else math.nan
        res.metrics["P_block"]        = p_block
        res.metrics["arrivals_total"] = self.attempted
        res.metrics["blocked"]        = self.blocked

        return res

# ──────────────────────────────────────────────────────────────────────
# 2.  Helper – one experiment with given k and capacity
# ──────────────────────────────────────────────────────────────────────

def simulate_k(base: Parameters, *, k: int, K: int,
               seed: int | None, bins: int = 80):
    """
    Run one IPP/M/1/K experiment and return
      • wq … numpy array of waiting-time samples
      • qs … numpy array of queue-length samples (time average)
      • st … dict with headline statistics
    """

    # ---------------------------------------------------------------
    # 1.  Build a super-position of k independent IPPs
    # ---------------------------------------------------------------
    master = random.Random(seed)
    rngs   = [random.Random(master.getrandbits(32)) for _ in range(k)]
    ipps   = [IPPArrivalProcess(base, rng=r) for r in rngs]

    class _Super:
        def __init__(self, streams):
            self._s = streams
            self._next = [ip.next_arrival(0.0) for ip in streams]

        def next_arrival(self, now: float):
            i = int(np.argmin(self._next))
            t = self._next[i]
            self._next[i] = self._s[i].next_arrival(t)
            return t

    # ---------------------------------------------------------------
    # 2.  Run the finite-capacity simulator
    # ---------------------------------------------------------------
    sim = FiniteCapacityIPPQueueSimulator(
              base, capacity=K,
              metrics=[BusyPeriodMetric()], log_events=True
          )
    sim.arrivals     = _Super(ipps)
    sim.next_arrival = sim.arrivals.next_arrival(sim.t)

    results = sim.run()                       # SimulationResults obj.

    # ---------------------------------------------------------------
    # 3.  Raw samples
    # ---------------------------------------------------------------
    wq = results.wait_times
    qs = np.asarray([                       # time-average queue length
           ev.queue_len if hasattr(ev, "queue_len") else ev["queue_len"]
           for ev in results.event_log
         ])

    # ---------------------------------------------------------------
    # 4.  Headline statistics
    # ---------------------------------------------------------------
    p_block = results.blocked / results.total_arrivals
    rho_th  = k * base.pi_on * base.lambda_on / base.mu

    stats_dict = {
        "k": k,
        "K": K,
        "rho_th":  rho_th,
        "rho_sim": results.utilisation,
        "E[Wq]":   wq.mean(),
        "Lq_time_avg": qs.mean(),
        "p_block": p_block,
    }

    return wq, qs, stats_dict

# ──────────────────────────────────────────────────────────────────────
# New CLI that supports *lists* for every numeric parameter
# ──────────────────────────────────────────────────────────────────────
import itertools, argparse
from typing import List

def _int_list(xs: List[str])   -> List[int]:   return [int(x)   for x in xs]
def _float_list(xs: List[str]) -> List[float]: return [float(x) for x in xs]

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "IPP/M/1/K simulator – finite capacity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # global options ---------------------------------------------------
    ap.add_argument("--seed", type=int, default=None, help="master RNG seed")
    ap.add_argument("-K", "--capacity",
                type=int, nargs="+", required=True, metavar="CAP",
                help="system capacity K (one or more)")

    ap.add_argument("-k", type=int, nargs="+", default=[3])
    ap.add_argument("--omega1", type=float, nargs="+", default=[0.2])
    ap.add_argument("--omega2", type=float, nargs="+", default=[0.3])
    ap.add_argument("--lam-on", dest="lam_on",
                    type=float, nargs="+", default=[1.5])
    ap.add_argument("--mu", type=float, nargs="+", default=[2.0])
    ap.add_argument("--horizon", type=float, default=1e5)
    ap.add_argument("--bins", type=int, default=80)

    return ap
# ---------------------------------------------------------------------

def run_grid(args):
    """
    Cartesian-product driver: runs once for every combination of
    {K, k, ω1, ω2, λON, μ}.  If len(k)==1 you effectively get the old
    single-run mode.
    """
    param_space = itertools.product(
        args.capacity, args.k, args.omega1, args.omega2, args.lam_on, args.mu
    )

    import pandas as pd
    rows = []
    for (K, k, w1, w2, lam_on, mu) in param_space:
        print(f"\n>>> Simulating  K={K}  k={k}  ω1={w1}  ω2={w2}"
              f"λON={lam_on}  μ={mu}")
        base = Parameters(
            omega_off_to_on=w1,
            omega_on_to_off=w2,
            lambda_on=lam_on,
            mu=mu,
            horizon=args.horizon,
            seed=args.seed,
        )
        wq, qs, st = simulate_k(base, k=k, K=K,
                                seed=args.seed, bins=args.bins)
        rows.append({**st, "K": K, "omega1": w1, "omega2": w2,
                     "lam_on": lam_on, "mu": mu})

    df = pd.DataFrame(rows)
    keep = ["K", "k", "omega1", "omega2", "lam_on", "mu",
            "rho_th", "rho_sim", "E[Wq]", "Lq_time_avg", "p_block"]
    print("\\n———— Summary over all runs ————")
    print(df[keep].to_string(index=False,
                             float_format=lambda x: f"{x:10.4f}"))
    print("———————————————————————————")

# ---------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    run_grid(args)

if __name__ == "__main__":        # plug into your existing file OR separate
    main()