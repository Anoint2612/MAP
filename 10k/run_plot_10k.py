#!/usr/bin/env python3
# run_plot_long.py
# Driver: compiles the above two programs and runs experiments (steps=10000).
# Uses N values divisible by 12 so domain decomposition has no remainder for 12 cores.

import os, subprocess, re, statistics, csv
import matplotlib.pyplot as plt
import psutil

def detect_cpu():
    phys = psutil.cpu_count(logical=False)
    logi = psutil.cpu_count(logical=True)
    print("\n=== CPU DETECTION ===")
    print(f"Physical cores : {phys}")
    return phys, logi

physical, logical = detect_cpu()

# Use physical cores by default (set True to use hyperthreads)
USE_HWT = False
if USE_HWT:
    mpirun_flags = ["--use-hwthread-cpus", "--bind-to", "hwthread", "--map-by", "hwthread"]
    max_procs = logical
else:
    mpirun_flags = ["--bind-to", "core", "--map-by", "core"]
    max_procs = physical

# Process list limited to max_procs
base_procs = [1, 2, 4, 6, 8, 10, 12]
procs = [p for p in base_procs if p <= max_procs]
print(f"\nMPI process list auto-selected = {procs}\n")

# Lattice sizes divisible by 12 (12 * {1024,2048,4096,8192})
lattice_sizes = [12*1024, 12*2048, 12*4096, 12*8192]   # [12288, 24576, 49152, 98304]
repeats = 3

# Compile fresh separate programs (won't touch your originals)
print("Compiling h_serial_10k.cpp -> h_serial_10k")
subprocess.run(["g++", "-O3", "h_serial_10k.cpp", "-o", "h_serial_10k"], check=True)
print("Compiling h_parallel_10k.cpp -> h_parallel_10k")
subprocess.run(["mpic++", "-O3", "h_parallel_10k.cpp", "-o", "h_parallel_10k"], check=True)

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"

time_regex = re.compile(r"Rank\s+\d+\s*\|\s*time\s*=\s*([\d.]+)\s*s")
serial_regex = re.compile(r"Serial runtime:\s*([\d.]+)")

# storage
rows = [["N","procs","serial_time_s","parallel_time_s_median","speedup"]]

plt.figure(figsize=(14,5))
plt.subplot(1,2,1); plt.title("Runtime vs Processes"); plt.xlabel("Processes"); plt.ylabel("Time (s)")
plt.subplot(1,2,2); plt.title("Speedup vs Processes"); plt.xlabel("Processes"); plt.ylabel("Speedup")

for N in lattice_sizes:
    print(f"\n=== N = {N} ===\n")

    # serial median
    stimes = []
    for _ in range(repeats):
        p = subprocess.run(["./h_serial_10k", str(N)], capture_output=True, text=True, env=env)
        out = p.stdout + p.stderr
        m = serial_regex.search(out)
        if not m:
            raise RuntimeError(f"No serial time parsed for N={N}, output:\n{out[:1000]}")
        stimes.append(float(m.group(1)))
    serial_time = statistics.median(stimes)
    print(f"Serial median time = {serial_time:.8f} s")

    runtimes = []
    speedups = []
    valid_p = []

    for pcount in procs:
        trial_times = []
        for _ in range(repeats):
            cmd = ["mpirun"] + mpirun_flags + ["-np", str(pcount), "./h_parallel_10k", str(N)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            out = r.stdout + r.stderr

            matches = time_regex.findall(out)
            if matches:
                times = [float(x) for x in matches]
                trial_times.append(max(times))
            else:
                fallback = re.findall(r"([\d.]+)\s*s", out)
                if fallback:
                    trial_times.append(float(fallback[-1]))
                    print(f"⚠ fallback time for N={N} p={pcount}")
                else:
                    print(f"⚠ no timing for N={N} p={pcount}; raw output first 300 chars:\n{out[:300]}")

        if not trial_times:
            print(f"⚠ skipping p={pcount} (no data)")
            continue

        median_rt = statistics.median(trial_times)
        sp = serial_time / median_rt if median_rt > 0 else float('nan')
        print(f"p={pcount}: median_time={median_rt:.8f}s  speedup={sp:.3f}")

        runtimes.append(median_rt)
        speedups.append(sp)
        valid_p.append(pcount)
        rows.append([N, pcount, serial_time, median_rt, sp])

    if valid_p:
        plt.subplot(1,2,1); plt.plot(valid_p, runtimes, marker='o', label=f"N={N}")
        plt.subplot(1,2,2); plt.plot(valid_p, speedups, marker='o', label=f"N={N}")
    else:
        print(f"No parallel data for N={N}")

# finalize
plt.subplot(1,2,1); plt.legend(); plt.grid(True)
plt.subplot(1,2,2); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("runtime_speedup_10k.png", dpi=300)
print("\nSaved runtime_speedup_10k.png")

with open("runtimes_speedups_10k.csv", "w", newline="") as f:
    writer = csv.writer(f); writer.writerows(rows)
print("Saved runtimes_speedups_10k.csv")
