# IPP / M / 1 Queue Simulator & Diagnostics

This repo simulates an **Interrupted-Poisson‐Process (IPP) arrival
stream feeding an M / M / 1 server** and provides a battery of
burstiness and queue-performance diagnostics.

The main entry point is **`ipp_main.py`**; the heavy lifting is done in
`ipp_queue_sim.py`.

---

## 1 · Quick start

```bash
# create a fresh environment (recommended):
conda create -n ippq python=3.11
conda activate ippq

# install dependencies
pip install numpy scipy matplotlib pandas

# run a single IPP/M/1 experiment
python ipp_main.py single --omega1 0.2 --omega2 0.3 --lam-on 1.5 --mu 2.0
All plots are written to a folder called figures/ (auto-created).

# 2 Command-line overview

python ipp_main.py <subcommand> [options]

subcommands:
  single   simulate one IPP source
  multi    superpose k identical IPPs (one or several k values)
  diag     arrival-stream diagnostics only (no queue)
2.1 Common options
(these appear on every sub-command)

flag	meaning	default
--omega1	OFF → ON rate ω₁ (s⁻¹)	0.2
--omega2	ON → OFF rate ω₂ (s⁻¹)	0.3
--lam-on	arrival rate in ON state λ<sub>ON</sub> (s⁻¹)	1.5
--mu	service rate μ (s⁻¹)	2.0
--horizon	simulation time horizon (s)	1e5
--seed	RNG seed (int)	None (time-based)

2.2 single
Extra flags	effect
--verify-mm1	also run a “near-Poisson’’ case and compare to the M/M/1 formula
--diag	run arrival-stream CV-test + hazard + ACF plots (hazard.png, acf.png)
--h2-test	Kolmogorov–Smirnov fit of IPP gaps to a 2-phase hyper-exponential (writes h2_hist.png)

2.3 multi
Extra flags	effect
-k 1 2 4	list one or more k values; one simulation per k
--bins N	histogram bin count (default 80)
--dep-diag	extra departure-stream & busy-period diagnostics

          → writes `dep_pdf_k<k>_mu<μ>.png`, `busy_pdf_k<k>_mu<μ>.png`
--plot | deprecated (plots are always written, two per k)

Outputs per run:

figures/wait_times_k<k>_...png – waiting-time histogram

figures/queue_len_k<k>_...png – queue-length PMF
and some statistics that are printed to the console

2.4 diag
Arrival-stream diagnostics only (no queue).
Options:

flag	effect
--diag-samples N	sample size (default 100 000)
--h2-test	add H₂ KS-test & histogram

# 3 Examples

# A. fixed μ, sweep k = 1..4, plus departure/busy diagnostics
python ipp_main.py multi -k 1 2 3 4 --mu 2.0 --dep-diag

# B. constant ρ = 0.8 (μ scaled inside a shell loop)
for k in 1 2 4 8; do
    mu=$(python - <<PY "$k"
import sys
k=int(sys.argv[1]); pi_on=0.2/(0.2+0.3); lam_on=1.5
print(k*pi_on*lam_on/0.8)
PY
)
    python ipp_main.py multi -k $k --mu $mu
done

# C. burstiness sweep at k = 3
python ipp_main.py multi -k 3 --omega1 0.05 --omega2 0.05
python ipp_main.py multi -k 3 --omega1 0.2  --omega2 0.3
python ipp_main.py multi -k 3 --omega1 5.0  --omega2 5.0
4 · Interpreting the output
Console table – one row per k; compare rho_th (theory) to
rho_sim (observed), and inspect means / variances of queue length and waiting times
​

PNG files – automatically named with all key parameters so runs
never overwrite each other
(figures/wait_times_k2_mu2p0_w10p2_w20p3_lam1p5.png, etc.).

Departure diagnostics (when --dep-diag is used) –
check dep_pdf_*.png for a near-exponential shape (CV²≈1) when
ρ < 1; busy-period pngs should match Exp(μ − λ<sub>eff</sub>)
unless the queue is unstable.

# 5 Project structure
.
├─ ipp_main.py          ← CLI driver (this file is what you run)
├─ ipp_queue_sim.py     ← event-driven simulator + metrics plug-ins
├─ run_experiments.sh   ← (optional) batch script that reproduces report figures
└─ figures/             ← all plots are written here
