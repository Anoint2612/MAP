# ================================================
# Strong-scaling test with Amdahl analysis
# steps = 10000, N divisible by 12, repeats = 5
# ================================================
import os, subprocess, re, statistics, csv
import matplotlib.pyplot as plt
import psutil
import numpy as np

# ------------------------------------------------
# Detect cores
# ------------------------------------------------
def detect_cpu_info():
    physical = psutil.cpu_count(logical=False)
    logical  = psutil.cpu_count(logical=True)

    print("\n=== CPU DETECTION ===")
    print(f"Physical cores : {physical}")
    print(f"Logical cores  : {logical}")
    
    return physical, logical

physical_cores, logical_cores = detect_cpu_info()

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------
lattice_sizes = [12288, 24576, 49152, 98304]     # All divisible by 12
repeats = 5
steps = 10000

# p list: use only physical cores for accuracy
procs = [1, 2, 4, 6, 8, 10, 12]

mpirun_flags = ["--bind-to", "core", "--map-by", "core", "--report-bindings"]

# ------------------------------------------------
# Compile
# ------------------------------------------------
print("\nCompiling programs ...\n")
subprocess.run(["g++", "-O3", "h_serial.cpp", "-o", "h_serial_10k"])
subprocess.run(["mpic++", "-O3", "h_parallel.cpp", "-o", "h_parallel_10k"])

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["STEPS"] = str(steps)     # if you want to pass steps via environment

# Regex for parallel output
time_regex = re.compile(r"Rank\s+\d+\s*\|\s*time\s*=\s*([\d.]+)\s*s")

# ------------------------------------------------
# CSV output
# ------------------------------------------------
csv_rows = [["N","p","serial_time","parallel_time","speedup","f_estimated"]]

# ------------------------------------------------
# Amdahl fit helper
# ------------------------------------------------
def compute_parallel_fraction(S, p):
    """Return Amdahl parallel fraction f."""
    if S <= 1e-12:
        return 0.0
    return (1 - 1/S) / (1 - 1/p)

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
plt.figure(figsize=(14,5))

# Subplots for measured S(p)
plt.subplot(1,2,1)
plt.title("Measured Speedup vs Processes")
plt.xlabel("Processes")
plt.ylabel("Measured Speedup")

# Subplots for Amdahl predicted curves
plt.subplot(1,2,2)
plt.title("Amdahl Predicted Speedup")
plt.xlabel("Processes")
plt.ylabel("Predicted S(p)")

for N in lattice_sizes:
    print(f"\n===================== N = {N} =====================")

    # ------------------------------------------
    # SERIAL baseline
    # ------------------------------------------
    serial_times = []
    for _ in range(repeats):
        proc = subprocess.run(["./h_serial", str(N)],
                              capture_output=True, text=True, env=env)
        out = proc.stdout
        tmatch = re.findall(r"Serial runtime:\s*([\d.]+)", out)
        if tmatch:
            serial_times.append(float(tmatch[-1]))
    serial_time = statistics.median(serial_times)

    print(f"Serial median time = {serial_time:.8f}s")

    runtimes = []
    speedups = []
    valid_p = []
    f_list  = []

    # ------------------------------------------
    # PARALLEL runs
    # ------------------------------------------
    for pcount in procs:

        trial_times = []

        for _ in range(repeats):
            cmd = ["mpirun"] + mpirun_flags + \
                  ["-np", str(pcount), "./h_parallel", str(N)]

            pr = subprocess.run(cmd, capture_output=True, text=True, env=env)
            out = pr.stdout

            # Parse per-rank times
            matches = time_regex.findall(out)
            if matches:
                tt = [float(t) for t in matches]
                trial_times.append(max(tt))

        if not trial_times:
            print(f"⚠ No data for p={pcount}")
            continue

        med_time = statistics.median(trial_times)
        speedup = serial_time / med_time
        f_val = compute_parallel_fraction(speedup, pcount)

        print(f"p={pcount}: t={med_time:.8f}s  speedup={speedup:.3f}  f={f_val:.4f}")

        valid_p.append(pcount)
        runtimes.append(med_time)
        speedups.append(speedup)
        f_list.append(f_val)

        csv_rows.append([N, pcount, serial_time, med_time, speedup, f_val])

    # =====================================================
    # Plot measured S(p)
    # =====================================================
    plt.subplot(1,2,1)
    plt.plot(valid_p, speedups, marker='o', label=f"N={N}")

    # =====================================================
    # Fit a single parallel fraction f using largest p
    # =====================================================
    if f_list:
        f_fit = f_list[-1]     # f estimated from highest p
        print(f"Fitted parallel fraction f ≈ {f_fit:.4f}")

        # Compute predicted Amdahl curve
        p_vals = np.array(valid_p)
        S_pred = 1.0 / ((1 - f_fit) + f_fit / p_vals)

        plt.subplot(1,2,2)
        plt.plot(valid_p, S_pred, marker='o', label=f"N={N} (f={f_fit:.3f})")

# Finalize plots
plt.subplot(1,2,1); plt.legend(); plt.grid(True)
plt.subplot(1,2,2); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig("amdahl_analysis.png", dpi=300)
print("\nSaved plot amdahl_analysis.png")

# Write CSV
with open("amdahl_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Saved CSV amdahl_results.csv")
