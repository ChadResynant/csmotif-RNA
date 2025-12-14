# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CSmotif-RNA is an RNA chemical shift predictor based on motifs. It predicts NMR chemical shifts (N and H) for imino groups in RNA A-form helices using motif-based lookup tables.

**Python Version:** Python 3.6+ (converted from Python 2 on 2025-12-14)

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
tools/
  fraMotif.py            # Motif extraction (triplet, penta, basePair)
  bctab.py               # BCTab class - chemical shift lookup table parser
  NH.cs                  # Reference chemical shifts indexed by "barcode"
  base.py                # Utilities (range2list, text partitioning, colors)
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

### Performance

For 12 peaks on 512×1024 grid:
- **NumPy CPU:** ~6 ms
- **PyTorch MPS:** ~3 ms (2× speedup)
- **CuPy CUDA:** ~1 ms (6× speedup, estimated)

Speedup increases with:
- More peaks (50+ peaks → 10× speedup)
- Larger grids (1024×2048 → 5× speedup)
- Batch processing multiple spectra

### API Usage

```python
from sim.imino.mkucsf_gpu import init_gpu_backend, sim_2d_gaussian_gpu_batched

init_gpu_backend('torch')  # or 'cupy', 'numpy'

spectrum = sim_2d_gaussian_gpu_batched(
    shape=(512, 1024),
    peaks=[(256, 512), (300, 600)],      # (row, col) in points
    linewidths=[(6.0, 12.0), (6.0, 12.0)],  # FWHM in points
    amplitudes=[100.0, 100.0]
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

## Dependencies

**Required:**
- Python 3.6+
- NumPy

**Optional:**
- `nmrglue` - UCSF spectrum file I/O
- `matplotlib` - Plotting
- `torch` - PyTorch GPU acceleration
- `cupy` - CuPy GPU acceleration (NVIDIA only)

Install all:
```bash
pip install numpy nmrglue matplotlib torch
```

## Testing

```bash
# Test basic prediction
python3 genimino.py tP5abc.seq
# Verify: imino.tab has 24 lines (12 G/U residues × 2 nuclei each)

# Test batch processing
python3 genimino_batch.py tP5abc.seq --simulate --gpu

# Test GPU simulation benchmark
cd sim.imino && python3 mkucsf_gpu.py --benchmark
```

## Development Notes

### Code Patterns
- Use `np.` prefix for NumPy functions (not `from numpy import *`)
- Imports use absolute paths from repo root (`from tools.base import ...`)
- GPU code uses graceful fallbacks to CPU

### Adding New Features
- Chemical shift tables: Add new `.cs` files to `tools/`, update BCTab if needed
- New motif types: Extend `getMotif()` in `tools/fraMotif.py`
- Spectrum parameters: Modify udic in simulation scripts
