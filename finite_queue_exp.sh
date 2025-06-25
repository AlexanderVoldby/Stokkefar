#!/usr/bin/env bash
# ===============================================================
# run_finite_experiments.sh – IPP/M/1/K study for the report
# ===============================================================
source "C:/Users/Anders/anaconda3/etc/profile.d/conda.sh"
conda activate 02443
# ===============================================================
SEED=42          # reproducible master seed
BINS=100         # histogram resolution
H=1e5            # horizon (s) – change here if you need longer runs

# -------- block A:  K–blocking curve (single IPP) ----------------
python ipp_finite_queue.py -K 5 10 20 40            \
        -k 1                                 \
        --mu 1.0 --omega1 0.2 --omega2 0.3 --lam-on 1.5 \
        --horizon $H --bins $BINS --seed $SEED

# -------- block B:  burstiness sweep (same ρ, K=20) ---------------
python ipp_finite_queue.py -K 20                    \
        -k 1                                 \
        --mu 1.0                                    \
        --omega1 0.05 0.2 5.0                       \
        --omega2 0.05 0.3 5.0                       \
        --lam-on 1.5 --horizon $H --bins $BINS --seed $SEED

# -------- block C:  Palm–Khintchine smoothing ---------------------
python ipp_finite_queue.py -K 20                    \
        -k 1 3 6                              \
        --mu 1.0 --omega1 0.05 --omega2 0.05 --lam-on 1.5 \
        --horizon $H --bins $BINS --seed $SEED

# -------- block D:  constant-ρ sweep (ρ≈0.8, k=3) -----------------
python ipp_finite_queue.py -K 20                    \
        -k 3                                 \
        --mu 0.8 1.2 1.6 2.0 3.0 4.0                \
        --omega1 0.2 --omega2 0.3 --lam-on 1.5      \
        --horizon $H --bins $BINS --seed $SEED

# -------- block E:  overload stress test (k=4) --------------------
python ipp_finite_queue.py -K 20 40 80              \
        -k 4                                 \
        --mu 2.0 --omega1 0.2 --omega2 0.3 --lam-on 1.5 \
        --horizon $H --bins $BINS --seed $SEED
