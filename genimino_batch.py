#!/usr/bin/env python3
"""
Batch RNA imino chemical shift prediction with optional GPU-accelerated
spectrum simulation.

This script processes multiple RNA sequences in parallel and can generate
2D NMR spectra using GPU acceleration.

Usage:
    # Single sequence
    python genimino_batch.py sequences/rna1.seq

    # Multiple sequences
    python genimino_batch.py sequences/*.seq

    # With spectrum generation
    python genimino_batch.py --simulate sequences/*.seq

    # GPU-accelerated spectrum generation
    python genimino_batch.py --simulate --gpu sequences/*.seq

Example input file format (rna.seq):
    GGCAGUACCAAGUCGCGAAAGCGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGCC
    (((((((((..(((((....))))).(((((....)))))..))).....))))))
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import numpy as np

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.base import range2list
from tools.bctab import BCTab
from tools.fraMotif import getMotif

# Imino nuclei definitions
NUS = {'G': ('N1', 'H1'), 'U': ('N3', 'H3')}


def predict_chemical_shifts(seq: str, bdstr: str, resis: Optional[List[int]] = None,
                            cs_table: Optional[BCTab] = None) -> List[Dict]:
    """
    Predict imino chemical shifts for an RNA sequence.

    Parameters
    ----------
    seq : str
        RNA sequence (ACGU)
    bdstr : str
        Bracket-dot secondary structure notation
    resis : list of int, optional
        Custom residue numbering. If None, uses 1-based numbering.
    cs_table : BCTab, optional
        Pre-loaded chemical shift table. If None, loads default.

    Returns
    -------
    list of dict
        Predicted chemical shifts with keys: residue, resname, N_ppm, H_ppm, N_atom, H_atom
    """
    if cs_table is None:
        cs_table = BCTab(os.path.join(os.path.dirname(__file__), 'tools/NH.cs'))

    CS = cs_table.NH

    if resis is None:
        resis = list(range(1, len(seq) + 1))

    predictions = []

    for i, s in enumerate(seq):
        if s not in NUS:
            continue

        resi = resis[i]
        idx, mode = getMotif('triplet', i, seq, bdstr)

        if mode:
            nuN, nuH = NUS[s]
            bd = '-'.join(mode[:3])

            try:
                N = CS[bd][0]
                H = CS[bd][1]
                predictions.append({
                    'residue': resi,
                    'resname': s,
                    'N_ppm': N,
                    'H_ppm': H,
                    'N_atom': nuN,
                    'H_atom': nuH,
                    'barcode': bd
                })
            except KeyError:
                # Unsupported base pair
                pass

    return predictions


def process_sequence_file(filepath: str, resi_range: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """
    Process a single sequence file.

    Returns
    -------
    tuple
        (filepath, predictions)
    """
    with open(filepath) as f:
        lines = f.readlines()

    seq = lines[0].strip()
    bdstr = lines[1].strip()

    resis = None
    if resi_range:
        resis = range2list(resi_range)
        if len(resis) != len(seq):
            raise ValueError(f"Residue range length ({len(resis)}) doesn't match sequence length ({len(seq)})")

    predictions = predict_chemical_shifts(seq, bdstr, resis)

    return filepath, predictions


def write_shift_table(predictions: List[Dict], output_path: str):
    """Write predictions to tab-separated file."""
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred['resname']}\t{pred['residue']:5d}\t{pred['N_atom']:>5s}\t{pred['N_ppm']:8.2f}\n")
            f.write(f"{pred['resname']}\t{pred['residue']:5d}\t{pred['H_atom']:>5s}\t{pred['H_ppm']:8.2f}\n")


def simulate_spectrum_gpu(predictions: List[Dict], output_path: str,
                          shape: Tuple[int, int] = (512, 1024),
                          use_gpu: bool = True):
    """
    Simulate 2D NMR spectrum from predicted chemical shifts.

    Uses GPU acceleration if available.
    """
    # Import GPU module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim.imino'))

    try:
        from mkucsf_gpu import (init_gpu_backend, sim_2d_gaussian_gpu_batched,
                                 GPU_BACKEND)
    except ImportError:
        print("Warning: GPU module not available, using basic simulation")
        use_gpu = False

    if use_gpu:
        init_gpu_backend('torch' if use_gpu else 'numpy')

    # Spectrum parameters
    maxcsH, mincsH = 14.5, 10.0
    maxcsN, mincsN = 165.0, 140.0
    sw = maxcsH - mincsH
    carH = (mincsH + sw/2.0) * 600.0
    swH = sw * 1.2 * 600.0
    sw = maxcsN - mincsN
    carN = (mincsN + sw/2.0) * 60.8
    swN = sw * 1.2 * 60.8

    # Convert ppm to points
    def ppm_to_pts_N(ppm):
        hz = ppm * 60.8
        frac = (hz - carN + swN/2) / swN
        return frac * shape[0]

    def ppm_to_pts_H(ppm):
        hz = ppm * 600.0
        frac = (hz - carH + swH/2) / swH
        return frac * shape[1]

    peaks_pts = []
    linewidths = []
    amplitudes = []

    lw_15N = 6.0
    lw_1H = 12.0

    for pred in predictions:
        pts_15N = ppm_to_pts_N(pred['N_ppm'])
        pts_1H = ppm_to_pts_H(pred['H_ppm'])
        peaks_pts.append((pts_15N, pts_1H))
        linewidths.append((lw_15N, lw_1H))
        amplitudes.append(100.0)

    if use_gpu:
        spectrum = sim_2d_gaussian_gpu_batched(shape, peaks_pts, linewidths, amplitudes)
    else:
        # Basic NumPy simulation
        spectrum = np.zeros(shape, dtype=np.float32)
        row_coords = np.arange(shape[0], dtype=np.float32)
        col_coords = np.arange(shape[1], dtype=np.float32)
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks_pts, linewidths, amplitudes):
            row_sigma = row_lw * fwhm_to_sigma
            col_sigma = col_lw * fwhm_to_sigma
            row_gauss = np.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = np.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)
            spectrum += amp * np.outer(row_gauss, col_gauss)

    # Save spectrum
    try:
        import nmrglue as ng

        udic = {
            'ndim': 2,
            0: {'car': carN, 'complex': False, 'encoding': 'states',
                'freq': True, 'label': '15N', 'obs': 60.8,
                'size': shape[0], 'sw': swN, 'time': False},
            1: {'car': carH, 'complex': False, 'encoding': 'direct',
                'freq': True, 'label': '1H', 'obs': 600.0,
                'size': shape[1], 'sw': swH, 'time': False}
        }

        dic = ng.sparky.create_dic(udic)
        ng.sparky.write(output_path, dic, spectrum.astype('float32'), overwrite=True)

    except ImportError:
        # Fall back to numpy save
        np.save(output_path.replace('.ucsf', '.npy'), spectrum)


def main():
    parser = argparse.ArgumentParser(
        description='Batch RNA imino chemical shift prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('files', nargs='+', help='Sequence files to process')
    parser.add_argument('-r', '--range', help='Residue range (e.g., 1-50,60-100)')
    parser.add_argument('-o', '--output', help='Output directory (default: current)')
    parser.add_argument('--simulate', action='store_true',
                        help='Generate 2D spectrum for each sequence')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for spectrum simulation')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--json', action='store_true',
                        help='Output predictions as JSON')

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    results = []

    if args.jobs > 1 and len(args.files) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(process_sequence_file, f, args.range)
                for f in args.files
            ]
            for future in futures:
                results.append(future.result())
    else:
        # Sequential processing
        for filepath in args.files:
            results.append(process_sequence_file(filepath, args.range))

    # Output results
    for filepath, predictions in results:
        basename = Path(filepath).stem

        # Write shift table
        tab_path = output_dir / f"{basename}_imino.tab"
        write_shift_table(predictions, str(tab_path))
        print(f"Wrote: {tab_path} ({len(predictions)} peaks)")

        # Optional JSON output
        if args.json:
            import json
            json_path = output_dir / f"{basename}_imino.json"
            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Wrote: {json_path}")

        # Optional spectrum simulation
        if args.simulate and predictions:
            ucsf_path = output_dir / f"{basename}_sim.ucsf"
            simulate_spectrum_gpu(predictions, str(ucsf_path), use_gpu=args.gpu)
            print(f"Wrote: {ucsf_path}")

    print(f"\nProcessed {len(results)} sequence(s), {sum(len(p) for _, p in results)} total peaks")


if __name__ == '__main__':
    main()
