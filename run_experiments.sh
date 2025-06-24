#!/usr/bin/env bash
source "C:/Users/Anders/anaconda3/etc/profile.d/conda.sh"
conda activate 02443
# A-block
python ipp_main.py --seed 42 multi -k 1 2 3 4 --mu 2.0 --omega1 0.2 --omega2 0.3 --lam-on 1.5 --bins 100 --dep-diag

# B-block  (loop in shell or makefile)
for k in 1 2 4 8; do
    # --- compute μ(k) in a subshell -----------------------------------
    mu=$(python - "$k" <<'PY'
import sys, math
k = int(sys.argv[1])
pi_on = 0.2 / (0.2 + 0.3)      # π_ON
lam_on = 1.5
print(k * pi_on * lam_on / 0.8)
PY
)
    mu=$(echo "$mu")           # remove the trailing newline

    # --- run the simulation -------------------------------------------
    python ipp_main.py --seed 42 multi -k "$k" --mu "$mu" \
        --omega1 0.2 --omega2 0.3 --lam-on 1.5 --bins 100 --dep-diag
done

# C-block (three runs)
python ipp_main.py --seed 42 multi -k 3 --mu 2.0 --omega1 0.05 --omega2 0.05 --lam-on 1.5 --dep-diag
python ipp_main.py --seed 42 multi -k 3 --mu 2.0 --omega1 0.2  --omega2 0.3  --lam-on 1.5 --dep-diag
python ipp_main.py --seed 42 multi -k 3 --mu 2.0 --omega1 5.0  --omega2 5.0  --lam-on 1.5 --dep-diag
