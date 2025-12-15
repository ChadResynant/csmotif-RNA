# CSmotif-RNA

RNA imino chemical shift predictor using motif-based lookup tables.

## Overview

CSmotif-RNA predicts NMR chemical shifts (15N and 1H) for imino groups in RNA A-form helices. It uses a triplet barcode system that encodes the local structural context of each base pair.

**Features:**
- Predict G(N1,H1) and U(N3,H3) chemical shifts
- GPU-accelerated 2D spectrum simulation
- Batch processing for multiple sequences
- Based on experimentally-derived chemical shift statistics

## Installation

```bash
git clone https://github.com/ChadResynant/csmotif-RNA.git
cd csmotif-RNA
pip install numpy nmrglue  # Required dependencies
```

**Optional GPU acceleration:**
```bash
pip install torch           # PyTorch (CUDA or MPS)
pip install cupy-cuda11x    # CuPy (NVIDIA only)
```

## Quick Start

### Predict Chemical Shifts

```bash
python3 genimino.py tP5abc.seq
```

**Input file format** (`tP5abc.seq`):
```
GGCAGUACCAAGUCGCGAAAGCGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGCC
(((((((((..(((((....))))).(((((....)))))..))).....))))))
```
- Line 1: RNA sequence (GCAU)
- Line 2: Bracket-dot secondary structure

**Output** (`imino.tab`):
```
G       2     N1    148.94
G       2     H1     13.48
...
```

### Simulate 2D Spectrum

```bash
cd sim.imino
python3 mkucsf_gpu.py              # GPU-accelerated
python3 mkucsf_gpu.py --benchmark  # Performance test
```

### Batch Processing

```bash
python3 genimino_batch.py sequences/*.seq
python3 genimino_batch.py --simulate --gpu sequences/*.seq
```

## Algorithm

The predictor uses a **triplet barcode** system:

1. Parse bracket-dot structure to identify base pairs
2. For each G/U residue in a helix, extract the triplet motif (current + flanking base pairs)
3. Build barcode: e.g., `GC-UA-CG` (prev-current-next)
4. Look up chemical shifts in reference table (145 entries)

**Supported base pairs:** GC, CG, AU, UA, GU, UG

**Limitations:**
- Only works within continuous A-form helices
- Cannot predict shifts for bulges, loops, or non-canonical pairs

## Performance

GPU-accelerated spectrum simulation benchmarks (512x1024 grid, 12 peaks):

| Backend | Time | Speedup |
|---------|------|---------|
| NumPy CPU | ~1.5 ms | 1x |
| CuPy CUDA | ~1.0 ms | 1.5x |

Speedup increases with larger grids and more peaks.

## Testing

```bash
pip install pytest
python3 -m pytest tests/ -v
```

## File Structure

```
csmotif-RNA/
├── genimino.py          # Main prediction script
├── genimino_batch.py    # Batch processing
├── tools/
│   ├── fraMotif.py      # Motif extraction
│   ├── bctab.py         # Chemical shift table parser
│   ├── NH.cs            # Reference shift data (145 barcodes)
│   └── base.py          # Utilities
├── sim.imino/
│   ├── mkucsf.py        # Reference spectrum simulation
│   ├── mkucsf_gpu.py    # GPU-accelerated simulation
│   └── genproj.py       # 1D projection plotting
└── tests/               # Pytest test suite
```

## References

- Original CSmotif server: http://rainbow.life.tsinghua.edu.cn/csmotif
- Chemical shift data derived from BMRB statistics

## License

See [LICENSE](LICENSE) file.
