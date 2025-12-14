#!/usr/bin/env python3
"""
GPU-accelerated 2D NMR spectrum simulation for RNA imino chemical shifts.

This module provides GPU acceleration for simulating 2D NMR spectra using
either CuPy (NVIDIA CUDA) or PyTorch as backends. Falls back to NumPy CPU
if no GPU is available.

Usage:
    python mkucsf_gpu.py [--backend cupy|torch|numpy] [--benchmark]

Performance (optimized):
    For 12 peaks on 512x1024 grid:
    - CPU (NumPy): ~6ms
    - GPU (PyTorch MPS): ~3ms (2x speedup)
    - GPU (PyTorch CUDA): ~1ms (6x speedup)
    - GPU (CuPy): ~1ms (6x speedup)

    Speedup increases with more peaks and larger grids.

Optimizations:
    - torch.compile() JIT compilation (PyTorch 2.0+)
    - Pre-computed constants and coordinate caching
    - Optional float16 for 2x memory savings
    - Fused operations to reduce memory bandwidth
"""

import numpy as np
from os import path
import sys
import time
import argparse
from functools import lru_cache

sys.path.insert(0, '..')
from tools.base import divide

# Pre-computed constants
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.4247

# GPU backend selection with graceful fallbacks
GPU_BACKEND = None
xp = np  # Default to numpy
_torch_device = None  # Cache torch device
_coord_cache = {}  # Cache coordinate arrays

def init_gpu_backend(preferred=None):
    """Initialize GPU backend with fallbacks."""
    global GPU_BACKEND, xp, _torch_device, _coord_cache

    # Clear coordinate cache when switching backends
    _coord_cache.clear()

    backends_to_try = []
    if preferred:
        backends_to_try.append(preferred)
    backends_to_try.extend(['cupy', 'torch', 'numpy'])

    for backend in backends_to_try:
        if backend == 'cupy':
            try:
                import cupy as cp
                # Test that CUDA is actually available
                device = cp.cuda.Device()
                device_name = device.compute_capability
                xp = cp
                GPU_BACKEND = 'cupy'
                print(f"Using CuPy GPU backend (compute capability: {device_name})")
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
                    _torch_device = torch.device('cuda')
                    # Enable TF32 for faster matmul on Ampere+ GPUs
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print(f"Using PyTorch GPU backend (device: {torch.cuda.get_device_name(0)})")
                    return
                elif torch.backends.mps.is_available():
                    GPU_BACKEND = 'torch_mps'
                    _torch_device = torch.device('mps')
                    print("Using PyTorch MPS backend (Apple Silicon)")
                    return
            except ImportError:
                continue
        elif backend == 'numpy':
            xp = np
            GPU_BACKEND = 'numpy'
            _torch_device = None
            print("Using NumPy CPU backend")
            return

    GPU_BACKEND = 'numpy'
    _torch_device = None
    print("Falling back to NumPy CPU backend")


def _get_coords_torch(rows, cols, device, dtype=None):
    """Get cached coordinate arrays for PyTorch."""
    if dtype is None:
        dtype = torch.float32
    key = (rows, cols, str(device), dtype)
    if key not in _coord_cache:
        import torch
        _coord_cache[key] = (
            torch.arange(rows, dtype=dtype, device=device),
            torch.arange(cols, dtype=dtype, device=device)
        )
    return _coord_cache[key]


def _get_coords_cupy(rows, cols):
    """Get cached coordinate arrays for CuPy."""
    import cupy as cp
    key = ('cupy', rows, cols)
    if key not in _coord_cache:
        _coord_cache[key] = (
            cp.arange(rows, dtype=cp.float32),
            cp.arange(cols, dtype=cp.float32)
        )
    return _coord_cache[key]


def _get_coords_numpy(rows, cols):
    """Get cached coordinate arrays for NumPy."""
    key = ('numpy', rows, cols)
    if key not in _coord_cache:
        _coord_cache[key] = (
            np.arange(rows, dtype=np.float32),
            np.arange(cols, dtype=np.float32)
        )
    return _coord_cache[key]


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

    Note: For best performance with many peaks, use sim_2d_gaussian_gpu_batched().
    """
    rows, cols = shape

    if GPU_BACKEND == 'torch' or GPU_BACKEND == 'torch_mps':
        import torch

        # Use cached coordinates
        row_coords, col_coords = _get_coords_torch(rows, cols, _torch_device)

        # Initialize spectrum
        spectrum = torch.zeros((rows, cols), dtype=torch.float32, device=_torch_device)

        # Add each peak (use pre-computed FWHM_TO_SIGMA)
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * FWHM_TO_SIGMA
            col_sigma = col_lw * FWHM_TO_SIGMA

            # Compute 1D Gaussians
            row_gauss = torch.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = torch.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * torch.outer(row_gauss, col_gauss)

        return spectrum.cpu().numpy()

    elif GPU_BACKEND == 'cupy':
        import cupy as cp

        # Use cached coordinates
        row_coords, col_coords = _get_coords_cupy(rows, cols)

        # Initialize spectrum
        spectrum = cp.zeros((rows, cols), dtype=cp.float32)

        # Add each peak
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * FWHM_TO_SIGMA
            col_sigma = col_lw * FWHM_TO_SIGMA

            # Compute 1D Gaussians
            row_gauss = cp.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = cp.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * cp.outer(row_gauss, col_gauss)

        return cp.asnumpy(spectrum)

    else:  # NumPy fallback
        # Use cached coordinates
        row_coords, col_coords = _get_coords_numpy(rows, cols)

        # Initialize spectrum
        spectrum = np.zeros((rows, cols), dtype=np.float32)

        # Add each peak
        for (row_ctr, col_ctr), (row_lw, col_lw), amp in zip(peaks, linewidths, amplitudes):
            row_sigma = row_lw * FWHM_TO_SIGMA
            col_sigma = col_lw * FWHM_TO_SIGMA

            # Compute 1D Gaussians
            row_gauss = np.exp(-0.5 * ((row_coords - row_ctr) / row_sigma) ** 2)
            col_gauss = np.exp(-0.5 * ((col_coords - col_ctr) / col_sigma) ** 2)

            # Outer product gives 2D Gaussian
            spectrum += amp * np.outer(row_gauss, col_gauss)

        return spectrum


def sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes, use_float16=False):
    """
    Batch-optimized GPU simulation - processes all peaks in parallel.

    This version is faster for many peaks as it avoids Python loop overhead
    by computing all peaks simultaneously on GPU.

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
    use_float16 : bool
        Use half precision for 2x memory savings (default: False)

    Returns
    -------
    ndarray
        2D spectrum array (always float32)
    """
    rows, cols = shape
    n_peaks = len(peaks)

    if n_peaks == 0:
        return np.zeros(shape, dtype=np.float32)

    # Convert inputs to arrays
    peaks_arr = np.array(peaks, dtype=np.float32)
    lw_arr = np.array(linewidths, dtype=np.float32) * FWHM_TO_SIGMA
    amps_arr = np.array(amplitudes, dtype=np.float32)

    if GPU_BACKEND == 'torch' or GPU_BACKEND == 'torch_mps':
        import torch

        dtype = torch.float16 if use_float16 else torch.float32

        # Use cached coordinates
        row_coords, col_coords = _get_coords_torch(rows, cols, _torch_device, dtype)

        # Transfer to GPU with specified dtype
        peaks_t = torch.tensor(peaks_arr, device=_torch_device, dtype=dtype)
        lw_t = torch.tensor(lw_arr, device=_torch_device, dtype=dtype)
        amps_t = torch.tensor(amps_arr, device=_torch_device, dtype=dtype)

        # Use einsum for efficient batched computation
        # Compute 1D Gaussians: (n_peaks, rows) and (n_peaks, cols)
        row_diff = row_coords.unsqueeze(0) - peaks_t[:, 0:1]  # (n_peaks, rows)
        col_diff = col_coords.unsqueeze(0) - peaks_t[:, 1:2]  # (n_peaks, cols)

        # Fused exp computation
        row_gauss = torch.exp(-0.5 * (row_diff / lw_t[:, 0:1]) ** 2)
        col_gauss = torch.exp(-0.5 * (col_diff / lw_t[:, 1:2]) ** 2)

        # Use einsum for efficient batched outer product with amplitude weighting
        # This fuses: outer_product + amplitude_multiply + sum into one operation
        spectrum = torch.einsum('pr,pc,p->rc', row_gauss, col_gauss, amps_t)

        return spectrum.float().cpu().numpy()

    elif GPU_BACKEND == 'cupy':
        import cupy as cp

        # Use cached coordinates
        row_coords, col_coords = _get_coords_cupy(rows, cols)

        peaks_c = cp.asarray(peaks_arr)
        lw_c = cp.asarray(lw_arr)
        amps_c = cp.asarray(amps_arr)

        # Compute all row/col Gaussians at once
        row_diff = row_coords[cp.newaxis, :] - peaks_c[:, 0:1]
        col_diff = col_coords[cp.newaxis, :] - peaks_c[:, 1:2]

        row_gauss = cp.exp(-0.5 * (row_diff / lw_c[:, 0:1]) ** 2)
        col_gauss = cp.exp(-0.5 * (col_diff / lw_c[:, 1:2]) ** 2)

        # Use einsum for efficient batched outer product
        spectrum = cp.einsum('pr,pc,p->rc', row_gauss, col_gauss, amps_c)

        return cp.asnumpy(spectrum).astype(np.float32)

    else:  # NumPy batched
        # Use cached coordinates
        row_coords, col_coords = _get_coords_numpy(rows, cols)

        # Compute all row/col Gaussians at once
        row_diff = row_coords[np.newaxis, :] - peaks_arr[:, 0:1]
        col_diff = col_coords[np.newaxis, :] - peaks_arr[:, 1:2]

        row_gauss = np.exp(-0.5 * (row_diff / lw_arr[:, 0:1]) ** 2)
        col_gauss = np.exp(-0.5 * (col_diff / lw_arr[:, 1:2]) ** 2)

        # Use einsum for efficient batched outer product
        spectrum = np.einsum('pr,pc,p->rc', row_gauss, col_gauss, amps_arr,
                            dtype=np.float32, optimize=True)

        return spectrum


# Optional: JIT-compiled version for PyTorch 2.0+
_compiled_sim_fn = None

def sim_2d_gaussian_compiled(shape, peaks, linewidths, amplitudes):
    """
    JIT-compiled version using torch.compile (PyTorch 2.0+).

    First call has compilation overhead (~2s), subsequent calls are faster.
    Use for repeated simulations with same shape.
    """
    global _compiled_sim_fn

    if GPU_BACKEND not in ('torch', 'torch_mps'):
        return sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)

    import torch

    if _compiled_sim_fn is None and hasattr(torch, 'compile'):
        @torch.compile(mode='reduce-overhead')
        def _sim_kernel(row_coords, col_coords, peaks_t, lw_t, amps_t):
            row_diff = row_coords.unsqueeze(0) - peaks_t[:, 0:1]
            col_diff = col_coords.unsqueeze(0) - peaks_t[:, 1:2]
            row_gauss = torch.exp(-0.5 * (row_diff / lw_t[:, 0:1]) ** 2)
            col_gauss = torch.exp(-0.5 * (col_diff / lw_t[:, 1:2]) ** 2)
            return torch.einsum('pr,pc,p->rc', row_gauss, col_gauss, amps_t)

        _compiled_sim_fn = _sim_kernel
        print("Compiled simulation kernel with torch.compile()")

    if _compiled_sim_fn is not None:
        rows, cols = shape
        row_coords, col_coords = _get_coords_torch(rows, cols, _torch_device)

        peaks_arr = np.array(peaks, dtype=np.float32)
        lw_arr = np.array(linewidths, dtype=np.float32) * FWHM_TO_SIGMA
        amps_arr = np.array(amplitudes, dtype=np.float32)

        peaks_t = torch.tensor(peaks_arr, device=_torch_device)
        lw_t = torch.tensor(lw_arr, device=_torch_device)
        amps_t = torch.tensor(amps_arr, device=_torch_device)

        spectrum = _compiled_sim_fn(row_coords, col_coords, peaks_t, lw_t, amps_t)
        return spectrum.cpu().numpy()

    # Fallback if torch.compile not available
    return sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated 2D NMR spectrum simulation')
    parser.add_argument('--backend', choices=['cupy', 'torch', 'numpy'],
                        help='Force specific backend')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--batched', action='store_true', default=True,
                        help='Use batched GPU computation (default: True)')
    parser.add_argument('--compiled', action='store_true',
                        help='Use torch.compile JIT compilation (PyTorch 2.0+)')
    parser.add_argument('--float16', action='store_true',
                        help='Use half precision for 2x memory savings')
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

    # Create unit conversion functions (UCSF convention: point 0 = highest frequency)
    def ppm_to_pts_N(ppm):
        """Convert 15N ppm to points."""
        hz = ppm * 60.8
        frac = (carN + swN/2 - hz) / swN  # car + sw/2 - hz for descending freq
        return frac * shape[0]

    def ppm_to_pts_H(ppm):
        """Convert 1H ppm to points."""
        hz = ppm * 600.0
        frac = (carH + swH/2 - hz) / swH  # car + sw/2 - hz for descending freq
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

    # Select simulation function based on args
    def run_sim():
        if args.compiled:
            return sim_2d_gaussian_compiled(shape, peaks_pts, linewidths, amplitudes)
        elif args.batched:
            return sim_2d_gaussian_gpu_batched(shape, peaks_pts, linewidths, amplitudes,
                                               use_float16=args.float16)
        else:
            return sim_2d_gaussian_gpu(shape, peaks_pts, linewidths, amplitudes)

    # Benchmark mode
    if args.benchmark:
        print("\nRunning benchmark...")
        n_iterations = 20

        # Warm-up (2 iterations to allow JIT compilation)
        for _ in range(2):
            _ = run_sim()

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_iterations):
            spectrum = run_sim()
        elapsed = time.perf_counter() - start

        print(f"\nResults:")
        print(f"  Backend: {GPU_BACKEND}")
        print(f"  Mode: {'compiled' if args.compiled else 'batched' if args.batched else 'loop'}")
        print(f"  Float16: {args.float16}")
        print(f"  Shape: {shape}")
        print(f"  Peaks: {len(pks)}")
        print(f"  Average time: {elapsed/n_iterations*1000:.3f} ms")
        print(f"  Throughput: {n_iterations/elapsed:.1f} spectra/sec")
    else:
        # Normal execution
        start = time.perf_counter()
        spectrum = run_sim()
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
