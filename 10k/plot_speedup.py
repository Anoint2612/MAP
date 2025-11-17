#!/usr/bin/env python3
# run_runtime_speedup.py
# Produce runtime and speedup plots (no Amdahl). Uses existing h_serial.cpp / h_parallel_10k.cpp

import os
import subprocess
import re
import statistics
import csv
import psutil
import matplotlib.pyplot as plt
import numpy as np
from shutil import which


groups = {
    "small": [1024, 2048, 4096, 8192],
    "large": [12288, 24576, 49152]
}
repeats = 7
timeout_sec = 300   # timeout for each run

candidate_procs = [1, 2, 4, 6, 8, 10, 12, 16]

# mpirun flags (core binding)
mpirun_flags = ["--bind-to", "core", "--map-by", "core", "--report-bindings"]


def detect_cores():
    phys = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    print("\n=== CPU DETECTION ===")
    print(f"Physical cores : {phys}")
    return phys, logical

def compile_binaries():
    # Ensure compilers present
    if which("g++") is None or which("mpic++") is None:
        raise RuntimeError("g++ and/or mpic++ not found in PATH")
    print("Compiling h_serial_10k.cpp -> h_serial_10k")
    subprocess.run(["g++", "-O3", "h_serial_10k.cpp", "-o", "h_serial_10k"], check=True)
    print("Compiling h_parallel_10k.cpp -> h_parallel_10k")
    subprocess.run(["mpic++", "-O3", "h_parallel_10k.cpp", "-o", "h_parallel_10k"], check=True)
    print("Compilation done.\n")

def parse_serial_time(output):
    m = re.findall(r"Serial runtime:\s*([\d.]+)\s*s", output)
    if m:
        return float(m[-1])
    # fallback: any 'X s' (last numeric occurrence)
    m2 = re.findall(r"([\d.]+)\s*s", output)
    if m2:
        return float(m2[-1])
    return None

def parse_parallel_times(output):
    # collect per-rank "Rank R | time = X s" entries
    matches = re.findall(r"Rank\s+\d+\s*\|\s*time\s*=\s*([\d.]+)\s*s", output)
    if matches:
        return [float(x) for x in matches]
    # fallback: any numeric "X s"
    any_times = re.findall(r"([\d.]+)\s*s", output)
    if any_times:
        return [float(any_times[-1])]
    return []

# ---------------------------
# Main
# ---------------------------
def main():
    phys, logical = detect_cores()
    procs = [p for p in candidate_procs if p <= phys]
    if not procs:
        procs = [1]
    print("MPI process list auto-selected =", procs, "\n")

    compile_binaries()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    for group_name, lattice_sizes in groups.items():
        print(f"\n=== GROUP: {group_name.upper()} ===\n")
        csv_file = f"runtimes_{group_name}.csv"
        csv_rows = [["N", "p", "serial_median_s", "parallel_median_s", "speedup"]]

        # prepare figure
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title(f"Runtime vs Processes ({group_name})")
        plt.xlabel("Processes")
        plt.ylabel("Runtime (s)")

        plt.subplot(1,2,2)
        plt.title(f"Speedup vs Processes ({group_name})")
        plt.xlabel("Processes")
        plt.ylabel("Speedup")

        for N in lattice_sizes:
            print(f"--- N = {N} ---")
            # serial baseline
            serial_times = []
            for r in range(repeats):
                try:
                    p = subprocess.run(["./h_serial_10k", str(N)], capture_output=True, text=True, env=env, timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    print(f"Serial run timeout for N={N} (repeat {r+1})")
                    continue
                t = parse_serial_time(p.stdout + p.stderr)
                if t is None:
                    print("Warning: couldn't parse serial time. Output head:\n", (p.stdout + p.stderr)[:400])
                else:
                    serial_times.append(t)
            if not serial_times:
                print(f"ERROR: no serial times for N={N}, skipping")
                continue
            serial_med = statistics.median(serial_times)
            print(f"Serial median time = {serial_med:.8f} s")

            # parallel runs across procs
            valid_p = []
            par_runtimes = []
            speedups = []

            for pcount in procs:
                trial_times = []
                for r in range(repeats):
                    cmd = ["mpirun"] + mpirun_flags + ["-np", str(pcount), "./h_parallel_10k", str(N)]
                    try:
                        pr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)
                    except subprocess.TimeoutExpired:
                        print(f"Parallel run timeout N={N} p={pcount} repeat={r+1}")
                        continue
                    times = parse_parallel_times(pr.stdout + pr.stderr)
                    if times:
                        trial_times.append(max(times))  
                if not trial_times:
                    print(f"⚠ No data for p={pcount} (N={N})")
                    continue
                med_par = statistics.median(trial_times)
                S = serial_med / med_par if med_par > 0 else float('nan')

                print(f"p={pcount}: median={med_par:.8f}s  speedup={S:.3f}")

                valid_p.append(pcount)
                par_runtimes.append(med_par)
                speedups.append(S)

                csv_rows.append([N, pcount, serial_med, med_par, S])

            # plot this N
            if valid_p:
                plt.subplot(1,2,1)
                plt.plot(valid_p, par_runtimes, marker='o', label=f"N={N}")
                plt.subplot(1,2,2)
                plt.plot(valid_p, speedups, marker='o', label=f"N={N}")

            print("")

        # finalize plots
        plt.subplot(1,2,1); plt.legend(); plt.grid(True)
        plt.subplot(1,2,2); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plotname = f"runtime_speedup_{group_name}.png"
        plt.savefig(plotname, dpi=300)
        print(f"Saved plot {plotname}")

        # write CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Saved CSV {csv_file}\n")

    print("All groups done.")

if __name__ == "__main__":
    main()
