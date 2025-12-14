# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CSmotif-RNA is an RNA chemical shift predictor based on motifs. It predicts NMR chemical shifts (N and H) for imino groups in RNA A-form helices using motif-based lookup tables.

**Note:** This is Python 2 code (uses `print` statements, `map()` returns lists, backtick string conversion).

## Usage

### Predict Imino Chemical Shifts

```bash
# Basic usage (residue numbering starts at 1)
python genimino.py <sequence_bracket-dot_file>

# With custom residue range
python genimino.py tP5abc.seq 130-145,150-153,158-193
```

**Input file format** (e.g., `tP5abc.seq`):
```
GGCAGUACCAAGUCGCGAAAGCGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGCC
(((((((((..(((((....))))).(((((....)))))..))).....))))))
```
- Line 1: RNA sequence
- Line 2: Bracket-dot secondary structure (only canonical base pairs: GC, CG, UA, AU, GU, UG)

**Output:** `imino.tab` - predicted chemical shifts for G(N1,H1) and U(N3,H3)

### Simulate 2D Spectrum

```bash
cd sim.imino
./clean.sh           # Remove previous outputs
python mkucsf.py     # Generate sim.ucsf (requires nmrglue)
```

## Architecture

```
genimino.py          # Main prediction script
tools/
  fraMotif.py        # Motif extraction (triplet, penta, basePair)
  bctab.py           # BCTab class - chemical shift lookup table parser
  NH.cs              # Reference chemical shifts indexed by "barcode" (triplet motif)
  base.py            # Utilities (range2list, text partitioning, ANSI colors)
sim.imino/
  mkucsf.py          # Spectrum simulation using nmrglue
  genproj.py         # 1D projection plotting
  tpl.save           # Sparky save file template
```

**Key concepts:**
- **Barcode:** Triplet motif identifier like "GC-UA-CG" (current base pair flanked by neighbors)
- **Motif types:** `basePair`, `triplet` (3 consecutive base pairs), `penta` (5 consecutive)
- The predictor only works for residues within continuous A-form helices (no bulges/loops)

## Chemical Shift Reference Data

The `tools/NH.cs` lookup table contains 145 triplet barcode entries with experimentally-derived chemical shifts:
- **N range:** ~142-163 ppm (imino nitrogen)
- **H range:** ~10-14.6 ppm (imino proton)
- **Format:** `Barcode  N_shift  H_shift` (space-separated)

**Imino nuclei by residue type:**
| Residue | Nitrogen | Proton |
|---------|----------|--------|
| G | N1 | H1 |
| U | N3 | H3 |

## Known Issues

### Import Path Problems in sim.imino/
The `mkucsf.py` script has broken imports:
```python
from common.base import divide  # Should be: from tools.base import divide
from sparky import *            # External dependency, not included
```

To run spectrum simulation, you need:
1. External `sparky` module (for Sparky save file generation)
2. External `mplot` module (for `genproj.py` plotting)
3. PYTHONPATH set to include parent directory

### Python 2 Only
Do NOT convert to Python 3 without explicit request. Key Python 2 patterns:
- `print 'text'` (no parentheses)
- `map()` returns list directly
- Backtick conversion: `` `resi` `` instead of `str(resi)`
- `reduce()` is built-in (not from functools)

## Testing

No automated tests exist. Manual testing:
```bash
# Test prediction
python genimino.py tP5abc.seq
# Verify imino.tab has 24 lines (12 G/U residues × 2 nuclei each)

# Test simulation (requires dependencies)
cd sim.imino && ./clean.sh && python mkucsf.py
```
