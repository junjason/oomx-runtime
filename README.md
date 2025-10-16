# Dynamic SoA Runtime & OO→Matrix Translator
**Patent-Pending — U.S. & PCT Applications Filed (2025)**  
**Author:** Sungmin (Jason) Jun

---

## 🧩 Overview
This repository demonstrates a **dynamic runtime architecture** that converts traditional **Object-Oriented (AoS)** simulation code into **matrix- and Structure-of-Arrays (SoA)** representations, executing them efficiently on multi-core CPUs and GPUs.

The system unifies **data layout transformation**, **runtime scheduling**, and **adaptive re-tiling** into one coherent engine.  
It began as a **Matrix-Based Game State Management System** and evolved into a **general-purpose Dynamic SoA Runtime** for parallel processors and high-volume transactional workloads.

> ⚙️ *In essence:* a V8-style managed runtime that continuously restructures memory for coalesced, divergence-free execution on data-parallel hardware.

---

## 🧠 Key Concepts

| Concept | Description |
|----------|--------------|
| **OO→Matrix Translation** | Converts compiled OO (Array-of-Structs) loops into a field-wise matrix or SoA form using LLVM/MLIR analysis. |
| **Dynamic SoA Runtime** | Monitors workload divergence (branch efficiency, active/inactive ratios) and re-tiles data at runtime with hysteresis control. |
| **Schema / Store / Sow Model** | A minimal ECS-like data engine: Schema defines fields, Store holds tiled SoA data, and Sow buffers deterministic writes. |
| **Pipeline API** | Functional pipeline (map/reduce) for per-tick or per-frame kernels, compatible with deterministic or in-place update modes. |
| **Cross-Domain Embodiments** | Gaming physics, particle systems, and financial micro-batch simulations all run on the same runtime. |

---

## ⚡ Why It Matters

Traditional simulation engines and financial batch systems still rely on **object trees and pointer-chasing loops**.  
This design breaks down on modern parallel hardware:

- GPUs execute thousands of threads in warps; branching and irregular memory layouts kill throughput.  
- CPUs with wide SIMD units suffer from low occupancy and cache misses.

The Dynamic SoA Runtime addresses both by:
- **Restructuring memory on the fly** to maintain warp coherence,
- **Compactly representing entities as columns** instead of heap objects,
- And **executing updates in tiled vectorized kernels**.

Early CPU benchmarks already show **30 %+ p95–p99 latency improvement** compared to baseline OO implementations.  
GPU benchmarks are in progress to quantify additional gains in branch efficiency and energy reduction.

---

## 🧪 Running the Benchmarks

### Prerequisites
- CMake ≥ 3.22  
- C++20 compiler (Clang or GCC)  
- (Optional) NVIDIA CUDA toolkit for GPU tests  

### Build
```bash
mkdir build && cd build
cmake .. && cmake --build . -j
Run (CPU)
bash
Copy code
# Static SoA
./build/tools/oomx-bench/oomx-bench --mode soa --sim gaming --n 10000 --ticks 100 --tile 8192

# Dynamic + index mode (with warmup to build idx off-clock)
./build/tools/oomx-bench/oomx-bench --mode soa --dynamic --sim gaming --n 10000 --ticks 100 --warmup 20 --tile 8192



📊 Example Output
ini
Copy code
sim=gaming mode=dyn n=1000000 ticks=600 tile=8192
mean=91.6us p95=127.9us p99=140.2us
Compared to static SoA:

ini
Copy code
mean=98.5us p95=144.2us p99=161.0us
