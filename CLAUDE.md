# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CSmotif-RNA is an RNA chemical shift predictor based on motifs. It predicts NMR chemical shifts (N and H) for imino groups in RNA A-form helices using motif-based lookup tables.

**Python Version:** Python 3.9+ (converted from Python 2 on 2025-12-14; `pyproject.toml` enforces `>=3.9`)

## Usage

### Basic Prediction

```bash
# Predict imino chemical shifts (residue numbering starts at 1)
python3 genimino.py <sequence_file>

# With custom residue range
python3 genimino.py tP5abc.seq 130-145,150-153,158-193
```

**Input file format** (e.g., `tP5abc.seq`):
```
GGCAGUACCAAGUCGCGAAAGCGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGCC
(((((((((..(((((....))))).(((((....)))))..))).....))))))
```
- Line 1: RNA sequence
- Line 2: Bracket-dot secondary structure (only canonical base pairs: GC, CG, UA, AU, GU, UG)

**Output:** `imino.tab` - predicted chemical shifts for G(N1,H1) and U(N3,H3)

### Batch Processing

```bash
# Process multiple sequences
python3 genimino_batch.py sequences/*.seq

# With spectrum generation
python3 genimino_batch.py --simulate sequences/*.seq

# GPU-accelerated spectrum generation
python3 genimino_batch.py --simulate --gpu sequences/*.seq

# Parallel processing with JSON output
python3 genimino_batch.py -j 4 --json sequences/*.seq
```

### Simulate 2D Spectrum

```bash
cd sim.imino
./clean.sh              # Remove previous outputs
python3 mkucsf.py       # Generate sim.ucsf using nmrglue

# GPU-accelerated simulation (2x faster on Apple Silicon, 10x+ on NVIDIA)
python3 mkucsf_gpu.py --benchmark
python3 mkucsf_gpu.py --backend numpy   # Force CPU
python3 mkucsf_gpu.py --backend torch   # Force PyTorch
python3 mkucsf_gpu.py --backend cupy    # Force CuPy (NVIDIA)
```

## Architecture

```
genimino.py              # Main prediction script
genimino_batch.py        # Batch processing with GPU support
shell_exec.py            # ONLY entry point for external process execution (frozen API)
platform_utils.py        # Cross-platform helpers (md5, memory, ping) — no shell
tools/
  fraMotif.py            # Motif extraction (triplet, penta, basePair)
  bctab.py               # BCTab class - chemical shift lookup table parser
  NH.cs                  # Reference chemical shifts indexed by "barcode"
  base.py                # Utilities (range2list, text partitioning, colors)
  ci/
    shell_hygiene_gate_hardmode.sh  # CI gate: zero-tolerance shell hazard check
sim.imino/
  mkucsf.py              # Spectrum simulation using nmrglue
  mkucsf_gpu.py          # GPU-accelerated spectrum simulation
  genproj.py             # 1D projection plotting (matplotlib)
  tpl.save               # Sparky save file template
```

## GPU Acceleration

The `mkucsf_gpu.py` module provides GPU-accelerated 2D NMR spectrum simulation.

### Supported Backends

| Backend | Device | Installation |
|---------|--------|--------------|
| PyTorch MPS | Apple Silicon | `pip install torch` |
| PyTorch CUDA | NVIDIA GPU | `pip install torch` |
| CuPy | NVIDIA GPU | `pip install cupy-cuda11x` |
| NumPy | CPU (fallback) | `pip install numpy` |

### Performance (Benchmarked)

For 12 peaks on 512×1024 grid:
- **NumPy CPU:** ~1.5 ms (optimized with einsum)
- **CuPy CUDA:** ~1.2 ms (RTX A2000)

For 500 peaks on 1024×2048 grid:
- **NumPy CPU:** ~14 ms
- **CuPy CUDA:** ~4 ms (3.3× speedup)

GPU speedup increases with problem size. Use `--benchmark` to test your system.

### Optimizations Applied

- **einsum operations:** Fused batched outer products reduce memory bandwidth
- **Coordinate caching:** Avoid repeated array allocations
- **Pre-computed constants:** FWHM-to-sigma at module level
- **TF32 enabled:** Faster matmul on Ampere+ GPUs
- **torch.compile:** Optional JIT compilation for PyTorch 2.0+ (`--compiled`)
- **float16 support:** Optional half precision (`--float16`)

### Command Line Options

```bash
cd sim.imino
python3 mkucsf_gpu.py --benchmark           # Run performance test
python3 mkucsf_gpu.py --backend cupy        # Force CuPy backend
python3 mkucsf_gpu.py --compiled            # Use torch.compile JIT
python3 mkucsf_gpu.py --float16             # Half precision (saves memory)
```

### API Usage

```python
from sim.imino.mkucsf_gpu import init_gpu_backend, sim_2d_gaussian_gpu_batched

init_gpu_backend('cupy')  # or 'torch', 'numpy'

# Standard usage
spectrum = sim_2d_gaussian_gpu_batched(
    shape=(512, 1024),
    peaks=[(256, 512), (300, 600)],      # (row, col) in points
    linewidths=[(6.0, 12.0), (6.0, 12.0)],  # FWHM in points
    amplitudes=[100.0, 100.0]
)

# Memory-efficient with float16
spectrum = sim_2d_gaussian_gpu_batched(
    shape=(512, 1024),
    peaks=peaks, linewidths=lws, amplitudes=amps,
    use_float16=True  # 2x memory savings
)
```

## Chemical Shift Reference Data

The `tools/NH.cs` lookup table contains 145 triplet barcode entries:
- **N range:** ~142-163 ppm (imino nitrogen)
- **H range:** ~10-14.6 ppm (imino proton)
- **Format:** `Barcode  N_shift  H_shift` (space-separated)

**Imino nuclei by residue type:**
| Residue | Nitrogen | Proton |
|---------|----------|--------|
| G | N1 | H1 |
| U | N3 | H3 |

**Key concepts:**
- **Barcode:** Triplet motif identifier like "GC-UA-CG" (current base pair flanked by neighbors)
- **Motif types:** `basePair`, `triplet` (3 consecutive base pairs), `penta` (5 consecutive)
- The predictor only works for residues within continuous A-form helices (no bulges/loops)

## Algorithm Flow

1. **Parse input:** Read sequence and bracket-dot structure from `.seq` file
2. **Build pairing map:** `pairing()` uses a stack to map each `(` to its matching `)` position
3. **For each G/U residue:** Call `getMotif('triplet', idx, seq, bdstr)`
   - Verify residue is base-paired (has partner in bracket-dot)
   - Check neighbors form continuous A-form helix (bp_cur, bp_pre, bp_nxt are stacked)
   - Return base pair identities: `[current_bp, prev_bp, next_bp]`
4. **Build barcode:** Join base pairs with `-` → e.g., "GC-UA-CG"
5. **Lookup shifts:** Query `NH.cs` table by barcode → returns (N_ppm, H_ppm)
6. **Output:** Write two lines per residue (N atom and H atom)

**Output format (`imino.tab`):**
```
ResName  ResNum  Atom    Shift
G            2     N1   148.94
G            2     H1    13.48
```

## Dependencies

**Required:**
- Python 3.6+
- NumPy

**Optional:**
- `nmrglue` - UCSF spectrum file I/O
- `matplotlib` - Plotting
- `torch` - PyTorch GPU acceleration
- `cupy` - CuPy GPU acceleration (NVIDIA only)

Install via pyproject.toml extras:
```bash
pip install -e ".[dev]"          # Core + pytest + nmrglue (recommended for development)
pip install -e ".[simulation]"   # Core + nmrglue
pip install -e ".[gpu]"          # Core + torch
```

## Testing

```bash
# Run full test suite (from repo root)
pytest

# Run a single test file
pytest tests/test_motif.py

# Run a single test by name
pytest -k "test_barcode_lookup"

# Smoke test: basic prediction (verify imino.tab has 24 lines = 12 residues × 2 nuclei)
python3 genimino.py tP5abc.seq

# Test GPU simulation benchmark
cd sim.imino && python3 mkucsf_gpu.py --benchmark
```

## Shell Hardening (CI-Enforced)

This repo enforces zero-tolerance shell hazard rules. CI blocks any new use of `os.system()`, `shell=True`, or `os.popen()`.

**Before pushing, run:**
```bash
./tools/ci/shell_hygiene_gate_hardmode.sh
```

**All external process calls must go through `shell_exec.py`:**
```python
from shell_exec import run_argv, run_bash, sh_quote, ExecKind

# Preferred: direct argv (no shell injection risk)
run_argv(["python3", script_path])

# Shell features only (pipes, redirects): use run_bash with sh_quote for untrusted values
run_bash(f"cat {sh_quote(path)} | wc -l")
```

**Never use:**
- `subprocess.run(..., shell=True)`
- `os.system(...)`
- `os.popen(...)`

For cross-platform helpers (file hashing, memory size, ping) that do NOT involve user-controlled data, use `platform_utils.py`.

## Development Notes

### Code Patterns
- Use `np.` prefix for NumPy functions (not `from numpy import *`)
- Imports use absolute paths from repo root (`from tools.base import ...`)
- GPU code uses graceful fallbacks to CPU
- `genimino_batch.py` (`predict_chemical_shifts()`) is the clean API; `genimino.py` is a legacy script

### Critical Gotchas

**`getMotif()` returns `(False, False)` on failure** — not `(None, None)`. Always check `if mode:` not `if mode is not None:`. Callers that check `idx` are checking the wrong variable (mode carries the base-pair identity; idx carries atom indices).

**`genimino.py` requires CWD = repo root** — it hardcodes `'./tools/NH.cs'`. Run it as `python3 genimino.py tP5abc.seq` from the repo root. `genimino_batch.py` uses `os.path.dirname(__file__)` and works from any directory.

**`BCTab` auto-detects nucleus type from value ranges** — `readTab()` inspects column 2 values to decide whether to populate `NH`, `CH`, or sugar `CH` dicts. When adding a new `.cs` file, the column 2 values must fall within one of the hardcoded ranges or the table will silently stay empty. Verify with `assert len(tab.NH) > 0` (or the relevant dict).

**`pairing()` is called inside every `getMotif()` call** — it recomputes the full bracket-dot parsing each time. For large batches, pre-compute `pairing(bdstr)` once and pass it directly if performance matters.

**PPM-to-points conversion** — the three-step process (PPM→Hz→fraction→points) is required for correct UCSF placement. Direct PPM×freq gives wrong positions. See `docs/lessons/FAILED_APPROACHES.md` and `genimino_batch.py:ppm_to_pts_N()` for the canonical implementation.

### Adding New Features
- Chemical shift tables: Add new `.cs` files to `tools/`, update BCTab if needed; verify the values fall in BCTab's auto-detected ranges
- New motif types: Extend `getMotif()` in `tools/fraMotif.py`
- Spectrum parameters: Modify udic in simulation scripts
