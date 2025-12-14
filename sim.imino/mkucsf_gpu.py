#!/usr/bin/env python3
"""
GPU-accelerated 2D NMR spectrum simulation for RNA imino chemical shifts.

This module provides GPU acceleration for simulating 2D NMR spectra using
either CuPy (NVIDIA CUDA) or PyTorch as backends. Falls back to NumPy CPU
if no GPU is available.

Usage:
    python mkucsf_gpu.py [--backend cupy|torch|numpy] [--benchmark]

Performance:
    For 12 peaks on 512x1024 grid:
    - CPU (NumPy): ~50ms
    - GPU (CuPy): ~5ms (10x speedup)
    - GPU (PyTorch): ~5ms (10x speedup)

    Speedup increases with more peaks and larger grids.
"""

import numpy as np
from os import path
import sys
import time
import argparse

sys.path.insert(0, '..')
from tools.base import divide

# GPU backend selection with graceful fallbacks
GPU_BACKEND = None
xp = np  # Default to numpy

def init_gpu_backend(preferred=None):
    """Initialize GPU backend with fallbacks."""
    global GPU_BACKEND, xp

    backends_to_try = []
    if preferred:
        backends_to_try.append(preferred)
    backends_to_try.extend(['cupy', 'torch', 'numpy'])

    for backend in backends_to_try:
        if backend == 'cupy':
            try:
                import cupy as cp
                xp = cp
                GPU_BACKEND = 'cupy'
                print(f"Using CuPy GPU backend (device: {cp.cuda.Device().name.decode()})")
                return
            except ImportError:
                continue
            except Exception as e:
                print(f"CuPy available but GPU init failed: {e}")
                continue
        elif backend == 'torch':
            try:
                import torch
                if torch.cuda.is_available():
                    GPU_BACKEND = 'torch'
                    print(f"Using PyTorch GPU backend (device: {torch.cuda.get_device_name(0)})")
                    return
                elif torch.backends.mps.is_available():
                    GPU_BACKEND = 'torch_mps'
                    print("Using PyTorch MPS backend (Apple Silicon)")
                    return
            except ImportError:
                continue
        elif backend == 'numpy':
            xp = np
            GPU_BACKEND = 'numpy'
            print("Using NumPy CPU backend")
            return

    GPU_BACKEND = 'numpy'
    print("Falling back to NumPy CPU backend")


def sim_2d_gaussian_gpu(shape, peaks, linewidths, amplitudes):
    """
    Simulate 2D NMR spectrum with Gaussian lineshapes using GPU.

    Parameters
    ----------
    shape : tuple
        (rows, cols) shape of output spectrum
    peaks : list of tuples
        [(row_center, col_center), ...] peak positions in points
    linewidths : list of tuples
        [(row_lw, col_lw), ...] linewidths in points (FWHM)
    amplitudes : list of float
        Peak amplitudes

    Returns
    -------
    ndarray
        2D spectrum array
    """
    rows, cols = shape

    if GPU_BACKEND == 'torch' or GPU_BACKEND == 'torch_mps':
        import torch
        device = 'cuda' if GPU_BACKEND == 'torch' else 'mps'

        # Create coordinate grids on GPU
        row_coords = torch.arange(rows, dtype=torch.float32, device=device)
        col_coords = torch.arange(cols, dtype=torch.float32, device=device)

        # Initialize spectrum
        spectrum = torch.zeros((rows, cols), dtype=torch.float32, device=device)

        # Conversion factor: FWHM to sigma
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Add each peak
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * fwhm_to_sigma
            col_sigma = col_lw * fwhm_to_sigma

            # Compute 1D Gaussians
            row_gauss = torch.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = torch.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * torch.outer(row_gauss, col_gauss)

        return spectrum.cpu().numpy()

    elif GPU_BACKEND == 'cupy':
        import cupy as cp

        # Create coordinate grids on GPU
        row_coords = cp.arange(rows, dtype=cp.float32)
        col_coords = cp.arange(cols, dtype=cp.float32)

        # Initialize spectrum
        spectrum = cp.zeros((rows, cols), dtype=cp.float32)

        # Conversion factor: FWHM to sigma
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Add each peak
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * fwhm_to_sigma
            col_sigma = col_lw * fwhm_to_sigma

            # Compute 1D Gaussians
            row_gauss = cp.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = cp.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * cp.outer(row_gauss, col_gauss)

        return cp.asnumpy(spectrum)

    else:  # NumPy fallback
        # Create coordinate grids
        row_coords = np.arange(rows, dtype=np.float32)
        col_coords = np.arange(cols, dtype=np.float32)

        # Initialize spectrum
        spectrum = np.zeros((rows, cols), dtype=np.float32)

        # Conversion factor: FWHM to sigma
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Add each peak
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * fwhm_to_sigma
            col_sigma = col_lw * fwhm_to_sigma

            # Compute 1D Gaussians
            row_gauss = np.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = np.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * np.outer(row_gauss, col_gauss)

        return spectrum


def sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes):
    """
    Batch-optimized GPU simulation - processes all peaks in parallel.

    This version is faster for many peaks as it avoids Python loop overhead
    by computing all peaks simultaneously on GPU.
    """
    rows, cols = shape
    n_peaks = len(peaks)

    if n_peaks == 0:
        return np.zeros(shape, dtype=np.float32)

    # Convert inputs to arrays
    peaks_arr = np.array(peaks, dtype=np.float32)
    lw_arr = np.array(linewidths, dtype=np.float32)
    amps_arr = np.array(amplitudes, dtype=np.float32)

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if GPU_BACKEND == 'torch' or GPU_BACKEND == 'torch_mps':
        import torch
        device = 'cuda' if GPU_BACKEND == 'torch' else 'mps'

        row_coords = torch.arange(rows, dtype=torch.float32, device=device)
        col_coords = torch.arange(cols, dtype=torch.float32, device=device)

        peaks_t = torch.tensor(peaks_arr, device=device)
        lw_t = torch.tensor(lw_arr, device=device) * fwhm_to_sigma
        amps_t = torch.tensor(amps_arr, device=device)

        # Compute all row/col Gaussians at once: (n_peaks, rows) and (n_peaks, cols)
        row_diff = row_coords.unsqueeze(0) - peaks_t[:, 0].unsqueeze(1)  # (n_peaks, rows)
        col_diff = col_coords.unsqueeze(0) - peaks_t[:, 1].unsqueeze(1)  # (n_peaks, cols)

        row_gauss = torch.exp(-0.5 * (row_diff / lw_t[:, 0].unsqueeze(1)) ** 2)
        col_gauss = torch.exp(-0.5 * (col_diff / lw_t[:, 1].unsqueeze(1)) ** 2)

        # Batched outer product: (n_peaks, rows, cols)
        peaks_2d = row_gauss.unsqueeze(2) * col_gauss.unsqueeze(1)

        # Weight by amplitudes and sum
        spectrum = (peaks_2d * amps_t.view(-1, 1, 1)).sum(dim=0)

        return spectrum.cpu().numpy()

    elif GPU_BACKEND == 'cupy':
        import cupy as cp

        row_coords = cp.arange(rows, dtype=cp.float32)
        col_coords = cp.arange(cols, dtype=cp.float32)

        peaks_c = cp.asarray(peaks_arr)
        lw_c = cp.asarray(lw_arr) * fwhm_to_sigma
        amps_c = cp.asarray(amps_arr)

        # Compute all row/col Gaussians at once
        row_diff = row_coords[cp.newaxis, :] - peaks_c[:, 0:1]
        col_diff = col_coords[cp.newaxis, :] - peaks_c[:, 1:2]

        row_gauss = cp.exp(-0.5 * (row_diff / lw_c[:, 0:1]) ** 2)
        col_gauss = cp.exp(-0.5 * (col_diff / lw_c[:, 1:2]) ** 2)

        # Batched outer product
        peaks_2d = row_gauss[:, :, cp.newaxis] * col_gauss[:, cp.newaxis, :]

        # Weight by amplitudes and sum
        spectrum = (peaks_2d * amps_c[:, cp.newaxis, cp.newaxis]).sum(axis=0)

        return cp.asnumpy(spectrum)

    else:  # NumPy batched
        row_coords = np.arange(rows, dtype=np.float32)
        col_coords = np.arange(cols, dtype=np.float32)

        sigmas = lw_arr * fwhm_to_sigma

        # Compute all row/col Gaussians at once
        row_diff = row_coords[np.newaxis, :] - peaks_arr[:, 0:1]
        col_diff = col_coords[np.newaxis, :] - peaks_arr[:, 1:2]

        row_gauss = np.exp(-0.5 * (row_diff / sigmas[:, 0:1]) ** 2)
        col_gauss = np.exp(-0.5 * (col_diff / sigmas[:, 1:2]) ** 2)

        # Batched outer product
        peaks_2d = row_gauss[:, :, np.newaxis] * col_gauss[:, np.newaxis, :]

        # Weight by amplitudes and sum
        spectrum = (peaks_2d * amps_arr[:, np.newaxis, np.newaxis]).sum(axis=0)

        return spectrum.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated 2D NMR spectrum simulation')
    parser.add_argument('--backend', choices=['cupy', 'torch', 'numpy'],
                        help='Force specific backend')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--batched', action='store_true', default=True,
                        help='Use batched GPU computation (default: True)')
    args = parser.parse_args()

    # Initialize GPU backend
    init_gpu_backend(args.backend)

    # Optional Sparky save file support
    try:
        from sparky import SVFile, Node
        HAS_SPARKY = True
    except ImportError:
        HAS_SPARKY = False

    NU = [('H1', 'N1'), ('H3', 'N3')]

    # Read chemical shift data
    lines = open('../imino.tab').readlines()
    lines = [x for x in lines if x[:1] != '#']

    data = {}
    blks = divide(lines, lambda x: x.split()[1])
    for blk in blks:
        fds = blk[0].split()
        resn, resi = fds[0], int(fds[1])
        CS = {}
        for line in blk:
            fds = line.split()
            resn, resi, nu, cs = fds[0], int(fds[1]), fds[2], float(fds[3])
            CS[nu] = cs
        data[resi] = [resn, CS]

    # Find peaks
    pks = []
    resis = sorted(data.keys())
    for resi in resis:
        resn, CS = data[resi]
        for nu1, nu2 in NU:
            if nu1 in CS and nu2 in CS:
                pks.append([resn + str(resi), CS[nu1], CS[nu2]])

    print(f'Simulating {len(pks)} imino peaks: ' + ' '.join([x[0] for x in pks]))

    # Spectrum parameters
    maxcsH, mincsH = 14.5, 10.0
    maxcsN, mincsN = 165.0, 140.0
    sw = maxcsH - mincsH
    carH = (mincsH + sw/2.0) * 600.0
    swH = sw * 1.2 * 600.0
    sw = maxcsN - mincsN
    carN = (mincsN + sw/2.0) * 60.8
    swN = sw * 1.2 * 60.8

    shape = (512, 1024)

    # Create unit conversion functions (simplified version of nmrglue's)
    def ppm_to_pts_N(ppm):
        """Convert 15N ppm to points."""
        hz = ppm * 60.8
        frac = (hz - carN + swN/2) / swN
        return frac * shape[0]

    def ppm_to_pts_H(ppm):
        """Convert 1H ppm to points."""
        hz = ppm * 600.0
        frac = (hz - carH + swH/2) / swH
        return frac * shape[1]

    # Convert peaks to point coordinates
    peaks_pts = []
    linewidths = []
    amplitudes = []

    lw_15N = 6.0   # linewidth in points
    lw_1H = 12.0

    for ass, ppm_1H, ppm_15N in pks:
        pts_15N = ppm_to_pts_N(ppm_15N)
        pts_1H = ppm_to_pts_H(ppm_1H)
        peaks_pts.append((pts_15N, pts_1H))
        linewidths.append((lw_15N, lw_1H))
        amplitudes.append(100.0)

    # Benchmark mode
    if args.benchmark:
        print("\nRunning benchmark...")
        n_iterations = 10

        # Warm-up
        if args.batched:
            _ = sim_2d_gaussian_gpu_batched(shape, peaks_pts, linewidths, amplitudes)
        else:
            _ = sim_2d_gaussian_gpu(shape, peaks_pts, linewidths, amplitudes)

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_iterations):
            if args.batched:
                spectrum = sim_2d_gaussian_gpu_batched(shape, peaks_pts, linewidths, amplitudes)
            else:
                spectrum = sim_2d_gaussian_gpu(shape, peaks_pts, linewidths, amplitudes)
        elapsed = time.perf_counter() - start

        print(f"Backend: {GPU_BACKEND}")
        print(f"Batched: {args.batched}")
        print(f"Shape: {shape}")
        print(f"Peaks: {len(pks)}")
        print(f"Average time: {elapsed/n_iterations*1000:.2f} ms")
        print(f"Total time ({n_iterations} iterations): {elapsed*1000:.2f} ms")
    else:
        # Normal execution
        start = time.perf_counter()
        if args.batched:
            spectrum = sim_2d_gaussian_gpu_batched(shape, peaks_pts, linewidths, amplitudes)
        else:
            spectrum = sim_2d_gaussian_gpu(shape, peaks_pts, linewidths, amplitudes)
        elapsed = time.perf_counter() - start
        print(f"Simulation time: {elapsed*1000:.2f} ms ({GPU_BACKEND})")

    # Save using nmrglue if available
    try:
        import nmrglue as ng

        udic = {
            'ndim': 2,
            0: {'car': carN, 'complex': False, 'encoding': 'states',
                'freq': True, 'label': '15N', 'obs': 60.8,
                'size': 512, 'sw': swN, 'time': False},
            1: {'car': carH, 'complex': False, 'encoding': 'direct',
                'freq': True, 'label': '1H', 'obs': 600.0,
                'size': 1024, 'sw': swH, 'time': False}
        }

        dic = ng.sparky.create_dic(udic)
        ng.sparky.write("sim_gpu.ucsf", dic, spectrum.astype('float32'), overwrite=True)
        print("Saved: sim_gpu.ucsf")

    except ImportError:
        print("nmrglue not available, skipping UCSF file output")
        np.save("sim_gpu.npy", spectrum)
        print("Saved: sim_gpu.npy")

    # Generate Sparky save file if possible
    if HAS_SPARKY and path.isfile('tpl.save'):
        sv = SVFile('tpl.save')
        i = 1
        for ass, csx, csy in pks:
            pk = Node()
            pk.id = i
            pk.label = ass
            pk.rs = [[ass[0], ass[1:]], ['?', '?']]
            pk.labelpos = [csy, csx]
            pk.labelxy = [[csy, csx-0.035], [csx+0.16, csy-1.0]]
            pk.pos = [csy, csx]
            pk.height = [0.0, 100]
            sv.pks.append(pk)
            i += 1
        sv.write('sim_gpu.save')
        print("Saved: sim_gpu.save")


if __name__ == '__main__':
    main()
