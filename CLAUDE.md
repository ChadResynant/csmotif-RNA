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
