# Session Log

## Updates Made
- Added `src/target_arch.h` with exact arch enum and flag parse for `-h100` and `-rtx5070`.
- Updated `scripts/benchmark.sh` to forward CLI args into benchmark binary.
- Updated `src/main.cu` to:
  - auto-detect GPU arch
  - accept arch override from CLI
  - print `# Target arch: ...`
  - shorten BF16 stdout table to match FP32 and TF32 table shape
- Updated benchmark outputs to write into arch-specific folders:
  - `results/h100/`
  - `results/rtx5070/`
- CSV now stores arch + GPU info per row.
- Updated FP32 dispatch to be arch-aware in:
  - `src/gemm_fp32_master.cu`
  - `src/gemm_fp32_r2z.cu`
  - `src/gemm_fp32_r2z_tc1.cu`
- Updated BF16 dispatch to be arch-aware in `src/gemm_bf16.cu`.
- Build passed after changes.
- RTX5070 benchmark passed with `./scripts/benchmark.sh -rtx5070`.
- Plotting now auto-discovers per-arch CSVs and produces self-contained HTML outputs with Plotly.
- H100 baseline results were copied back into `results/h100/` and plots were generated from the CSV.
- Added a PTX export mode via `scripts/benchmark.sh -ptx` that writes kernel PTX under `ptx/<gpu>/<compute>/`.
- Added `scripts/pull_runpod_artifacts.sh` to archive local `results/h100/` and `ptx/` before pulling fresh copies from Runpod.
- `scripts/benchmark.sh` now archives the previous live benchmark run into `results/archive/<arch>/<timestamp>/` and keeps only the newest 10 snapshots.

## Important Context
- Goal: cuBLAS-like or better BF16 and FP32 GEMM performance.
- Acceptable range: 80% to 90%+ of cuBLAS if better is not reachable.
- Target hardware:
  - H100 is main perf target.
  - RTX5070 is local dev box.
- Current FP32 goal from `CLAUDE.md`:
  - `naive` for tiny matrices
  - `r2z` for larger matrices
- BF16 remains separate baseline.
- TF32 / tensor-core path is still experimental.
- Do not reintroduce `FP32_VARIANT`.
- Benchmark stays square-matrix only.
- Keep `M * N` as selector key for now.
- `src/dd.cu` looks like duplicate TF32 code and is not in build.
- `h100.txt` and `rtx5070.txt` capture latest benchmark runs.
- PTX exports are generated from the kernel-bearing CUDA translation units in `src/`.

## Current Progress
- The benchmark harness now supports arch-aware runs, arch-specific result folders, and timestamped archive snapshots.
- The PTX export path is in place and can generate per-kernel PTX under `ptx/<gpu>/<compute>/`.
- The Runpod pull workflow is set up so local H100 and PTX artifacts can be refreshed without losing prior local copies.
- H100 baseline data has already been collected, copied back, and plotted.
- The current bottleneck is not data collection or reporting; it is kernel-side H100 tuning.

## H100 Optimization Plan
- Baseline takeaway:
  - H100 still has meaningful gap(s) versus cuBLAS on the custom paths.
  - The first priority is fixing underfilled or imbalanced mid-size launches, not broad refactors.
- Step 1: analyze the baseline gaps.
  - inspect the H100 plots for the largest runtime / throughput misses
  - separate BF16, FP32 CU-core, and FP32 TF32 reference behavior
  - note whether the issue is tiny-size overhead, mid-size occupancy, or large-size saturation
- Step 2: optimize BF16 first.
  - inspect launch shape and tile coverage for underfilled mid-size matrices
  - look for memory movement or occupancy bottlenecks before touching math structure
  - prefer one focused change at a time so benchmark deltas stay readable
- Step 3: optimize FP32 CUDA-core path.
  - keep `naive` for tiny matrices and `r2z` for the rest
  - check whether H100 needs a different size split than RTX5070
  - evaluate split-K / StreamK only if the mid-large gap stays material
- Step 4: keep TF32 secondary.
  - use it as a reference path, not the main optimization target
  - only invest there if it clarifies the H100 ceiling or exposes an obvious win
- Step 5: clean up after the tuning direction is clear.
  - remove dead duplicate TF32 code if it is not part of the active path
  - keep CSV and stdout formats stable while the kernels change

## Next Steps
- Inspect the H100 plots and rank the worst gaps by size and dtype.
- Start with BF16 on H100, since it likely has the highest ROI.
- Check whether the H100 BF16 config split is leaving mid-size matrices underutilized.
- If BF16 is mostly launch-geometry limited, adjust tile sizing or block coverage before changing the math pipeline.
- If BF16 looks structurally fine, move to FP32 `r2z` and compare the H100 cutoffs against the RTX5070 cutoffs.
- Only consider split-K / StreamK if the remaining H100 gap is still large after basic launch tuning.
- Keep TF32 as a reference path unless it exposes a clear H100 advantage.

## PTX Export Plan
- Default `-ptx` mode exports PTX for both H100 and RTX5070 targets.
- Passing `-h100` or `-rtx5070` restricts export to that GPU family only.
- PTX output is stored outside `results/` so it stays separate from benchmark artifacts.
- Only kernel-containing `.cu` files are exported; host-only wrapper files are skipped.

## Runpod Pull Workflow
- Use `scripts/pull_runpod_artifacts.sh HOST PORT` from the local machine.
- The script archives existing local `results/h100/` and `ptx/` copies before overwriting them.
- The `ptx/` archive lands in `ptx.archive/<timestamp>/` so it does not recurse into the live PTX tree.
- Default SSH key is `~/.ssh/id_ed25519`.
- Remote source path defaults to `~/cu_x_gemm`.

## Benchmark Archive Plan
- Live benchmark outputs remain in `results/h100/` and `results/rtx5070/`.
- Before each run, the script moves the prior `benchmark_results.csv` and `benchmark.log` into a timestamped archive snapshot.
- After a successful run, old archive snapshots beyond the newest 10 are pruned.

## H100 Run Workflow
- Yes, SSH only to run code, then copy results back.
- Best loop:
  - sync code to server
  - run benchmark remotely
  - copy back `results/benchmark_results.csv`
  - copy back `results/benchmark.log`
- Example remote run shape:
  - `ssh user@host 'cd /path/to/repo && ./scripts/benchmark.sh -h100'`
- Example copy-back shape:
  - `scp user@host:/path/to/repo/results/benchmark_results.csv ./h100_results.csv`
  - `scp user@host:/path/to/repo/results/benchmark.log ./h100.log`
- Keep remote runs non-interactive.
- If server cost is high, prefer one-shot SSH commands over shells.
- For easy analysis, keep these artifacts only:
  - CSV
  - log
  - optional short summary text
