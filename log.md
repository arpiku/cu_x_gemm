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

## H100 Optimization Plan
- Step 1: establish a clean H100 baseline.
  - rerun `./scripts/benchmark.sh -h100`
  - compare BF16, FP32 CU-core, and FP32 TF32 reference paths against the latest CSV
  - note the biggest size gaps versus cuBLAS
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
