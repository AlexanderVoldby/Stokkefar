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
# 3.  CLI driver (single or multi‑k)
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("IPP/M/1/K simulator – finite capacity")
    ap.add_argument("--seed", type=int, default=None, help="master RNG seed (None → random)")
    ap.add_argument("--capacity", "-K", type=int, required=True, help="system capacity (queue + service)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--omega1", type=float, default=0.2)
    common.add_argument("--omega2", type=float, default=0.3)
    common.add_argument("--lam-on", type=float, dest="lam_on", default=1.5)
    common.add_argument("--mu", type=float, default=2.0)
    common.add_argument("--horizon", type=float, default=1e5)
    common.add_argument("--bins", type=int, default=80)

    s_multi = sub.add_parser("multi", parents=[common], help="superpose one or more IPPs")
    s_multi.add_argument("-k", type=int, nargs="+", default=[3])
    s_multi.set_defaults(func=run_multi)

    s_one = sub.add_parser("single", parents=[common])
    s_one.add_argument("-k", type=int, default=1)
    s_one.set_defaults(func=run_single)
    return ap

# ----------------------------------------------------------------------

def run_single(args):
    base = Parameters(
        omega_off_to_on=args.omega1, omega_on_to_off=args.omega2,
        lambda_on=args.lam_on, mu=args.mu, horizon=args.horizon, seed=args.seed)
    simulate_k(base, k=args.k, K=args.capacity, seed=args.seed, bins=args.bins)

def run_multi(args):
    base = Parameters(
        omega_off_to_on=args.omega1, omega_on_to_off=args.omega2,
        lambda_on=args.lam_on, mu=args.mu, horizon=args.horizon, seed=args.seed)
    for k in args.k:
        simulate_k(base, k=k, K=args.capacity, seed=args.seed, bins=args.bins)

# ----------------------------------------------------------------------

def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
