# Failed Approaches — csmotif-RNA

Lessons from debugging sessions. Read this before attempting fixes in this repo.

---

## 2026-01-14: PPM-to-Points Coordinate Conversion (UCSF Format)

**Symptom:** Simulated peaks appear at wrong positions in `sim.ucsf`; spectrum looks shifted or mirrored relative to expected peak positions.

**Failed Attempt:** Directly multiplying PPM values by spectrometer frequency to get Hz, then using Hz as point coordinates.

**Why It Failed:** UCSF format stores the carrier frequency and sweep width in Hz. The correct conversion is:
1. Convert PPM → Hz: `hz = ppm * obs_freq`
2. Compute fractional position: `frac = (hz - car + sw/2) / sw`
3. Convert to points: `pts = frac * n_points`

Skipping step 2 (the carrier-relative fraction) places peaks relative to 0 Hz instead of relative to the spectral window center.

**Correct Solution:** Three-step PPM→Hz→fraction→points (commit `beb1e5c`). See `genimino_batch.py:ppm_to_pts_N()` and `ppm_to_pts_H()` for reference implementations.

**Prevention:** When adding new spectral windows, verify against `mkucsf.py` reference implementation using ZNCC > 0.999 (test in `test_integration.py:test_reference_vs_gpu_match`).

---

## 2025-12-14: Python 2 → Python 3 Conversion

**Symptom:** `print` as statement, `unicode`/`basestring` types, integer division, `dict.iteritems()` throughout original codebase.

**Failed Attempt:** Manual find-replace of `print x` → `print(x)` without checking for other Python 2 idioms.

**Why It Failed:** Missed integer division (`/` vs `//`), `dict.has_key()` calls, and implicit relative imports.

**Correct Solution:** Systematic sweep with `2to3` tool, followed by manual review of numeric operations to ensure division semantics were preserved.

**Prevention:** Run `python3 -W error::SyntaxWarning genimino.py` to catch remaining Python 2 idioms early.
