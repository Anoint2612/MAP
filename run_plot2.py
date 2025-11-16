import os, subprocess, re, statistics, csv
import matplotlib.pyplot as plt
import psutil

def detect_cpu_info():
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    print("\n=== CPU DETECTION ===")
    print(f"Physical cores : {physical}")

    return physical, logical

physical_cores, logical_cores = detect_cpu_info()

# ============================================
#  SETTINGS
# ============================================
lattice_sizes = [1024, 2048, 4096, 8192, 16384]
repeats = 3

USE_HYPERTHREADS = False   

if USE_HYPERTHREADS:
    max_procs = logical_cores
    mpirun_flags = ["--use-hwthread-cpus", "--bind-to", "hwthread", "--map-by", "hwthread"]
else:
    max_procs = physical_cores
    mpirun_flags = ["--bind-to", "core", "--map-by", "core"]

# ----- Generate MPI process list -----
base_list = [1, 2, 4, 6, 8, 10, 12]
procs = [p for p in base_list if p <= max_procs]

print(f"\nMPI process list auto-selected = {procs}\n")

# ============================================
#  COMPILE PROGRAMS
# ============================================
subprocess.run(["g++", "-O3", "h_serial.cpp", "-o", "h_serial"])
subprocess.run(["mpic++", "-O3", "h_parallel.cpp", "-o", "h_parallel"])

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"

# ============================================
#  CSV Setup
# ============================================
csv_rows = [["N","procs","serial_time_s","parallel_time_s","speedup"]]

# Regex for "Rank X | time = T s"
time_regex = re.compile(r"Rank\s+\d+\s*\|\s*time\s*=\s*([\d.]+)\s*s")

# ============================================
#  PLOT SETUP
# ============================================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.title("Runtime vs Processes")
plt.xlabel("Processes")
plt.ylabel("Time (s)")

plt.subplot(1,2,2)
plt.title("Speedup vs Processes")
plt.xlabel("Processes")
plt.ylabel("Speedup")

# ============================================
#  MAIN LOOP
# ============================================
for N in lattice_sizes:
    print(f"\n========== N = {N} ==========\n")

    # --------------------------------------
    # SERIAL RUN (median)
    # --------------------------------------
    serial_times = []
    for _ in range(repeats):
        p = subprocess.run(["./h_serial", str(N)], capture_output=True, text=True, env=env)
        out = p.stdout
        tmatch = re.findall(r"Serial runtime:\s*([\d.]+)", out)
        if tmatch:
            serial_times.append(float(tmatch[-1]))

    serial_time = statistics.median(serial_times)
    print(f"Serial median time = {serial_time:.8f} s")

    runtimes = []
    speedups = []
    valid_p = []

    # --------------------------------------
    # PARALLEL RUNS
    # --------------------------------------
    for pcount in procs:
        trial_times = []

        for _ in range(repeats):
            cmd = ["mpirun"] + mpirun_flags + ["-np", str(pcount), "./h_parallel", str(N)]
            pr = subprocess.run(cmd, capture_output=True, text=True, env=env)
            out = pr.stdout

            matches = time_regex.findall(out)

            if matches:
                times = [float(x) for x in matches]
                trial_times.append(max(times))  
            else:
                fallback = re.findall(r"([\d.]+)\s*s", out)
                if fallback:
                    trial_times.append(float(fallback[-1]))
                    print(f"⚠ Fallback used for N={N}, p={pcount}")
                else:
                    print(f"⚠ No timing detected for N={N}, p={pcount}")

        if not trial_times:
            print(f"⚠ Skipping p={pcount} (no data)")
            continue

        median_time = statistics.median(trial_times)
        sp = serial_time / median_time  

        runtimes.append(median_time)
        speedups.append(sp)
        valid_p.append(pcount)

        print(f"p={pcount}: time={median_time:.8f}s   speedup={sp:.3f}")

        csv_rows.append([N, pcount, serial_time, median_time, sp])

    # ---- plotting ----
    if valid_p:
        plt.subplot(1,2,1)
        plt.plot(valid_p, runtimes, marker='o', label=f"N={N}")

        plt.subplot(1,2,2)
        plt.plot(valid_p, speedups, marker='o', label=f"N={N}")

# ============================================
#  FINALIZE PLOTS
# ============================================
plt.subplot(1,2,1); plt.legend(); plt.grid(True)
plt.subplot(1,2,2); plt.legend(); plt.grid(True)
plt.tight_layout()

plt.savefig("runtime_and_speedup.png", dpi=300)
print("\nSaved plot runtime_and_speedup.png")

# ============================================
#  SAVE CSV
# ============================================
with open("runtimes_speedups.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Saved CSV runtimes_speedups.csv\n")
