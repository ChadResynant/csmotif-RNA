"""Tests for GPU-accelerated spectrum simulation (sim.imino/mkucsf_gpu.py)."""

import pytest
import numpy as np
import os
import sys

# Add sim.imino to path
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sim.imino"
))


class TestConstants:
    """Tests for pre-computed constants."""

    def test_fwhm_to_sigma(self):
        """Test FWHM to sigma conversion constant."""
        from mkucsf_gpu import FWHM_TO_SIGMA

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        # sigma = FWHM / (2 * sqrt(2 * ln(2)))
        expected = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        assert abs(FWHM_TO_SIGMA - expected) < 1e-15

    def test_fwhm_roundtrip(self):
        """Test that FWHM conversion produces correct Gaussian width."""
        from mkucsf_gpu import FWHM_TO_SIGMA

        fwhm = 10.0
        sigma = fwhm * FWHM_TO_SIGMA

        # Generate Gaussian and measure FWHM
        x = np.linspace(-50, 50, 10001)
        gauss = np.exp(-0.5 * (x / sigma) ** 2)

        # Find FWHM by finding where Gaussian crosses 0.5
        half_max = 0.5
        above_half = x[gauss >= half_max]
        measured_fwhm = above_half[-1] - above_half[0]

        assert abs(measured_fwhm - fwhm) < 0.01


class TestGPUBackend:
    """Tests for GPU backend initialization."""

    def test_backend_init(self):
        """Test GPU backend initializes without error."""
        from mkucsf_gpu import init_gpu_backend, GPU_BACKEND

        init_gpu_backend()

        from mkucsf_gpu import GPU_BACKEND
        assert GPU_BACKEND in ['cupy', 'torch', 'torch_mps', 'numpy']

    def test_numpy_fallback(self):
        """Test NumPy fallback works."""
        from mkucsf_gpu import init_gpu_backend, GPU_BACKEND

        init_gpu_backend('numpy')

        from mkucsf_gpu import GPU_BACKEND
        assert GPU_BACKEND == 'numpy'


class TestSimulation:
    """Tests for spectrum simulation functions."""

    @pytest.fixture
    def init_backend(self):
        """Initialize backend before tests."""
        from mkucsf_gpu import init_gpu_backend
        init_gpu_backend()

    def test_single_peak(self, init_backend):
        """Test simulation with single peak."""
        from mkucsf_gpu import sim_2d_gaussian_gpu

        shape = (64, 128)
        peaks = [(32, 64)]
        linewidths = [(4.0, 8.0)]
        amplitudes = [100.0]

        spectrum = sim_2d_gaussian_gpu(shape, peaks, linewidths, amplitudes)

        assert spectrum.shape == shape
        assert spectrum.dtype == np.float32

        # Peak should be near center
        max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        assert abs(max_idx[0] - 32) <= 1
        assert abs(max_idx[1] - 64) <= 1

        # Max value should be close to amplitude
        assert abs(np.max(spectrum) - 100.0) < 1.0

    def test_batched_single_peak(self, init_backend):
        """Test batched simulation with single peak."""
        from mkucsf_gpu import sim_2d_gaussian_gpu_batched

        shape = (64, 128)
        peaks = [(32, 64)]
        linewidths = [(4.0, 8.0)]
        amplitudes = [100.0]

        spectrum = sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)

        assert spectrum.shape == shape
        assert spectrum.dtype == np.float32

    def test_multiple_peaks(self, init_backend):
        """Test simulation with multiple peaks."""
        from mkucsf_gpu import sim_2d_gaussian_gpu_batched

        shape = (128, 256)
        peaks = [(32, 64), (64, 128), (96, 192)]
        linewidths = [(4.0, 8.0), (4.0, 8.0), (4.0, 8.0)]
        amplitudes = [100.0, 50.0, 75.0]

        spectrum = sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)

        assert spectrum.shape == shape

        # Check each peak region has signal
        assert spectrum[32, 64] > 50
        assert spectrum[64, 128] > 25
        assert spectrum[96, 192] > 35

    def test_empty_peaks(self, init_backend):
        """Test simulation with no peaks."""
        from mkucsf_gpu import sim_2d_gaussian_gpu_batched

        shape = (64, 128)
        peaks = []
        linewidths = []
        amplitudes = []

        spectrum = sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)

        assert spectrum.shape == shape
        assert np.all(spectrum == 0)

    def test_loop_vs_batched(self, init_backend):
        """Test that loop and batched methods produce same result."""
        from mkucsf_gpu import sim_2d_gaussian_gpu, sim_2d_gaussian_gpu_batched

        shape = (64, 128)
        peaks = [(20, 40), (40, 80), (50, 100)]
        linewidths = [(4.0, 8.0), (5.0, 10.0), (3.0, 6.0)]
        amplitudes = [100.0, 75.0, 50.0]

        spectrum_loop = sim_2d_gaussian_gpu(shape, peaks, linewidths, amplitudes)
        spectrum_batch = sim_2d_gaussian_gpu_batched(shape, peaks, linewidths, amplitudes)

        # Should be very close (floating point differences only)
        max_diff = np.max(np.abs(spectrum_loop - spectrum_batch))
        assert max_diff < 1e-4

    def test_float16_mode(self, init_backend):
        """Test float16 mode produces reasonable results."""
        from mkucsf_gpu import sim_2d_gaussian_gpu_batched, GPU_BACKEND

        # Only test if using torch backend
        if GPU_BACKEND not in ('torch', 'torch_mps'):
            pytest.skip("float16 test requires PyTorch backend")

        shape = (64, 128)
        peaks = [(32, 64)]
        linewidths = [(4.0, 8.0)]
        amplitudes = [100.0]

        spectrum_f32 = sim_2d_gaussian_gpu_batched(
            shape, peaks, linewidths, amplitudes, use_float16=False
        )
        spectrum_f16 = sim_2d_gaussian_gpu_batched(
            shape, peaks, linewidths, amplitudes, use_float16=True
        )

        # Results should be similar (float16 has lower precision)
        max_diff = np.max(np.abs(spectrum_f32 - spectrum_f16))
        assert max_diff < 1.0  # Allow more tolerance for float16


class TestCoordinateCaching:
    """Tests for coordinate array caching."""

    def test_cache_reuse(self):
        """Test that coordinate cache is reused."""
        from mkucsf_gpu import (
            init_gpu_backend, _coord_cache, _get_coords_numpy
        )

        init_gpu_backend('numpy')

        # Clear cache
        _coord_cache.clear()

        # First call should create cache entry
        coords1 = _get_coords_numpy(100, 200)
        assert ('numpy', 100, 200) in _coord_cache

        # Second call should return same objects
        coords2 = _get_coords_numpy(100, 200)
        assert coords1[0] is coords2[0]
        assert coords1[1] is coords2[1]


class TestEinsumFormula:
    """Tests for einsum batched outer product formula."""

    def test_einsum_matches_loop(self):
        """Test einsum formula matches explicit loop."""
        np.random.seed(42)

        n_peaks = 5
        rows, cols = 50, 100

        row_gauss = np.random.rand(n_peaks, rows).astype(np.float32)
        col_gauss = np.random.rand(n_peaks, cols).astype(np.float32)
        amps = np.random.rand(n_peaks).astype(np.float32) * 100

        # Loop implementation
        spectrum_loop = np.zeros((rows, cols), dtype=np.float32)
        for i in range(n_peaks):
            spectrum_loop += amps[i] * np.outer(row_gauss[i], col_gauss[i])

        # Einsum implementation
        spectrum_einsum = np.einsum('pr,pc,p->rc', row_gauss, col_gauss, amps)

        max_diff = np.max(np.abs(spectrum_loop - spectrum_einsum))
        assert max_diff < 1e-5
