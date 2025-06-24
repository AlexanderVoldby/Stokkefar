"""ipp_queue_sim.py
Simulation of a single‑server queue fed by an Interrupted Poisson Process (IPP).
--------------------------------------------------------------------------
Primary task for 02443 Stochastic Simulation project (see project description).

Model summary
-------------
Arrival process: Two‑state continuous‑time Markov chain (OFF→ON rate ω₁, ON→OFF rate ω₂).
                In the ON state arrivals occur according to a Poisson process with rate λ.
Service process: Exponential i.i.d. service times with rate μ (M‑server).
Queue discipline: FIFO, unlimited capacity, single server (M/M/1 when ω₁→∞, ω₂→∞).

Metrics collected
-----------------
* Average waiting time (E[W])
* Average number in system (E[L])
* Time‑dependent sample paths of queue length (for plotting, if desired)

The code is self‑contained: `pip install numpy` is the only external dependency.
Run the file directly to see a quick demonstration.

Author: ChatGPT‑o3
Date: 2025‑06‑20
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class IPPParameters:
    """Parameters defining the Interrupted Poisson Process."""

    lam: float  # Arrival intensity when in the ON state (λ)
    omega_on: float  # Transition rate OFF → ON (ω₁)
    omega_off: float  # Transition rate ON → OFF (ω₂)

    def stationary_on_prob(self) -> float:
        """Stationary probability of being in the ON state."""
        return self.omega_on / (self.omega_on + self.omega_off)

    def effective_lambda(self) -> float:
        """Long‑run average arrival rate of the IPP."""
        return self.lam * self.stationary_on_prob()


@dataclass
class SimulationResult:
    mean_waiting_time: float
    mean_queue_length: float
    utilisation: float
    arrivals: int
    departures: int
    times: List[float] = field(repr=False)
    queue_lengths: List[int] = field(repr=False)

    def __post_init__(self):
        # Improve repr brevity
        self.times = self.times[:5] + ["…"] if len(self.times) > 6 else self.times
        self.queue_lengths = self.queue_lengths[:5] + ["…"] if len(self.queue_lengths) > 6 else self.queue_lengths


class IPPQueueSimulator:
    """Discrete‑event simulator for an M/IPP/1 queue (unlimited buffer)."""

    def __init__(
        self,
        ipp: IPPParameters,
        mu: float,
        t_end: float = 10_000.0,
        rng: random.Random | None = None,
    ):
        self.ipp = ipp
        self.mu = mu
        self.t_end = t_end
        self.rng = rng or random.Random()

        # State variables
        self.t: float = 0.0  # simulation clock
        self.state_on: bool = self._initial_on_state()  # IPP state
        self.queue: List[float] = []  # arrival times of customers waiting/being served
        self.server_busy: bool = False

        # Stats
        self.arrivals = 0
        self.departures = 0
        self.total_waiting_time = 0.0
        self.area_num_in_system = 0.0  # for time‑integrated L(t)
        self.last_event_time = 0.0
        self.sample_times: List[float] = []
        self.sample_qlens: List[int] = []

        # Schedule first events
        self.next_state_change = self.t + self._exp(self._current_omega())
        self.next_arrival = self._schedule_next_arrival()
        self.next_departure = math.inf

    # ---------------------------------------------------------------------
    # Core simulation loop
    # ---------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Run the simulation until `t_end` and return aggregated statistics."""
        while self.t < self.t_end:
            # Determine next event
            next_event = min(self.next_arrival, self.next_departure, self.next_state_change)

            # Update time‑area integral for L(t)
            self._accumulate_area(next_event)

            # Advance clock
            self.t = next_event

            if self.t == self.next_state_change:
                self._handle_state_change()
            elif self.t == self.next_arrival:
                self._handle_arrival()
            else:  # departure
                self._handle_departure()

            # (Optional) store sample path – decimate for memory efficiency
            if len(self.sample_times) < 10_000:  # crude decimation
                self.sample_times.append(self.t)
                self.sample_qlens.append(len(self.queue))

        # Wrap up statistics
        mean_wait = self.total_waiting_time / self.departures if self.departures else 0.0
        mean_L = self.area_num_in_system / self.t_end
        rho = self.ipp.effective_lambda() / self.mu

        return SimulationResult(
            mean_waiting_time=mean_wait,
            mean_queue_length=mean_L,
            utilisation=rho,
            arrivals=self.arrivals,
            departures=self.departures,
            times=self.sample_times,
            queue_lengths=self.sample_qlens,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_state_change(self):
        self.state_on = not self.state_on
        self.next_state_change = self.t + self._exp(self._current_omega())
        # Reschedule arrival (may become enabled or disabled)
        self.next_arrival = self._schedule_next_arrival()

    def _handle_arrival(self):
        self.arrivals += 1
        # Record arrival time to compute waiting time later
        self.queue.append(self.t)
        # If server idle, start service immediately
        if not self.server_busy:
            self.server_busy = True
            self.next_departure = self.t + self._exp(self.mu)
        # Schedule next arrival
        self.next_arrival = self._schedule_next_arrival()

    def _handle_departure(self):
        self.departures += 1
        arrival_time = self.queue.pop(0)
        self.total_waiting_time += self.t - arrival_time
        if self.queue:
            # Start next service
            self.next_departure = self.t + self._exp(self.mu)
        else:
            self.server_busy = False
            self.next_departure = math.inf

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _initial_on_state(self) -> bool:
        """Draw initial IPP state according to stationary distribution."""
        return self.rng.random() < self.ipp.stationary_on_prob()

    def _current_omega(self) -> float:
        """Return the appropriate transition rate from the current state."""
        return self.ipp.omega_off if self.state_on else self.ipp.omega_on

    def _exp(self, rate: float) -> float:
        """Sample an exponential(rate) random variate."""
        return self.rng.expovariate(rate)

    def _schedule_next_arrival(self) -> float:
        if not self.state_on:
            return math.inf
        return self.t + self._exp(self.ipp.lam)

    def _accumulate_area(self, next_t: float):
        """Accumulate area under L(t) between last_event_time and next_t."""
        dt = next_t - self.last_event_time
        self.area_num_in_system += dt * len(self.queue)
        self.last_event_time = next_t


# ----------------------------------------------------------------------
# Quick demonstration and sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters chosen so that λ_eff / μ ≈ 0.8 (moderate utilisation)
    ipp = IPPParameters(lam=5.0, omega_on=2.0, omega_off=3.0)
    mu = 8.0

    sim = IPPQueueSimulator(ipp, mu, t_end=50_000.0, rng=random.Random(42))
    res = sim.run()

    print("Simulation finished:")
    print(f"  Arrivals          : {res.arrivals}")
    print(f"  Departures        : {res.departures}")
    print(f"  Utilisation ρ     : {res.utilisation:.3f}")
    print(f"  Mean queue length : {res.mean_queue_length:.3f}")
    print(f"  Mean waiting time : {res.mean_waiting_time:.3f} time‑units")

    # Expected M/M/1 mean queue length for λ_eff, μ
    lam_eff = ipp.effective_lambda()
    mm1_L = lam_eff / (mu - lam_eff)
    print(f"  M/M/1 benchmark   : E[L] = {mm1_L:.3f} (λ_eff={lam_eff:.3f})")
