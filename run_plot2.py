import os
import subprocess
import matplotlib.pyplot as plt
import re

# ======== SETTINGS ========
N = 1024              # system size in your C++ code
procs = [1, 2, 4, 8, 10]
# ===========================

# Compile codes
subprocess.run(["mpic++", "-O3", "h_parallel.cpp", "-o", "h_parallel"])
subprocess.run(["g++", "-O3", "h_serial.cpp", "-o", "h_serial"])

# Run serial baseline
serial_output = subprocess.run(["./h_serial"], capture_output=True, text=True).stdout
serial_time = float(re.findall(r"([\d.]+)\s*s", serial_output)[0])
print(f"✅ Serial runtime: {serial_time:.4f} s")

runtimes = []
speedups = []
valid_procs = []

# Environment: ensure 1 thread per MPI rank
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

for p in procs:
    print(f"\n🚀 Running with {p} MPI processes...")

    # Run MPI program with proper binding
    cmd = [
        "mpirun",
        "--bind-to", "core", "--map-by", "core",
        "--report-bindings",
        "-np", str(p),
        "./h_parallel"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for {p} processes. Skipping.")
        continue

    print(f"[Output for {p} processes]:\n{output.strip()}\n")

    # Extract runtime safely
    matches = re.findall(r"([\d.]+)\s*s", output)
    if not matches:
        print(f"⚠️ No valid runtime found for {p} processes — check output above.")
        continue

    time = float(matches[-1])  # last time in output
    valid_procs.append(p)
    runtimes.append(time)
    speedups.append(serial_time / time)

# ===== Plot results =====
plt.figure(figsize=(12, 5))

# Runtime plot
plt.subplot(1, 2, 1)
plt.plot(valid_procs, runtimes, marker='o', linestyle='-')
plt.xlabel("Number of MPI Processes")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Processes")

# Speedup plot
plt.subplot(1, 2, 2)
plt.plot(valid_procs, speedups, marker='o', linestyle='-')
plt.xlabel("Number of MPI Processes")
plt.ylabel("Speedup")
plt.title("Speedup vs Processes")

plt.tight_layout()
plt.savefig("hamiltonian_speedup_runtime.png", dpi=300)
print("✅ Graph saved as 'hamiltonian_speedup_runtime.png'")
