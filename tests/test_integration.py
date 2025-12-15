"""Integration tests for full prediction and simulation pipeline."""

import pytest
import numpy as np
import os
import sys
import subprocess
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPredictionPipeline:
    """Integration tests for chemical shift prediction."""

    def test_genimino_runs(self, tp5abc_data):
        """Test genimino.py runs without error."""
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run in temp directory to avoid polluting repo
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy necessary files
            shutil.copy(os.path.join(repo_dir, "tP5abc.seq"), tmpdir)
            shutil.copytree(os.path.join(repo_dir, "tools"), os.path.join(tmpdir, "tools"))
            shutil.copy(os.path.join(repo_dir, "genimino.py"), tmpdir)

            result = subprocess.run(
                [sys.executable, "genimino.py", "tP5abc.seq"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, f"genimino.py failed: {result.stderr}"
            assert os.path.exists(os.path.join(tmpdir, "imino.tab"))

    def test_prediction_output_format(self, tp5abc_data):
        """Test prediction output has correct format."""
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(os.path.join(repo_dir, "tP5abc.seq"), tmpdir)
            shutil.copytree(os.path.join(repo_dir, "tools"), os.path.join(tmpdir, "tools"))
            shutil.copy(os.path.join(repo_dir, "genimino.py"), tmpdir)

            subprocess.run(
                [sys.executable, "genimino.py", "tP5abc.seq"],
                cwd=tmpdir,
                capture_output=True
            )

            with open(os.path.join(tmpdir, "imino.tab")) as f:
                lines = f.readlines()

            # Should have pairs of lines (N and H for each residue)
            assert len(lines) % 2 == 0

            for line in lines:
                parts = line.split()
                assert len(parts) == 4  # ResName, ResNum, Atom, Shift

                resname = parts[0]
                assert resname in ['G', 'U']  # Only G and U have imino

                atom = parts[2]
                assert atom in ['N1', 'H1', 'N3', 'H3']

                shift = float(parts[3])
                if atom in ['N1', 'N3']:
                    assert 140 <= shift <= 165  # 15N range
                else:
                    assert 10 <= shift <= 15  # 1H range


class TestSimulationPipeline:
    """Integration tests for spectrum simulation."""

    @pytest.fixture
    def setup_sim_dir(self):
        """Set up simulation directory for tests."""
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_dir = os.path.join(repo_dir, "sim.imino")

        # Generate imino.tab first
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(os.path.join(repo_dir, "tP5abc.seq"), tmpdir)
            shutil.copytree(os.path.join(repo_dir, "tools"), os.path.join(tmpdir, "tools"))
            shutil.copy(os.path.join(repo_dir, "genimino.py"), tmpdir)

            subprocess.run(
                [sys.executable, "genimino.py", "tP5abc.seq"],
                cwd=tmpdir,
                capture_output=True
            )

            # Copy imino.tab to repo root for simulation
            shutil.copy(
                os.path.join(tmpdir, "imino.tab"),
                os.path.join(repo_dir, "imino.tab")
            )

        return sim_dir

    def test_mkucsf_gpu_runs(self, setup_sim_dir):
        """Test mkucsf_gpu.py runs without error."""
        result = subprocess.run(
            [sys.executable, "mkucsf_gpu.py", "--backend", "numpy"],
            cwd=setup_sim_dir,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"mkucsf_gpu.py failed: {result.stderr}"

    def test_gpu_output_valid_ucsf(self, setup_sim_dir):
        """Test GPU simulation produces valid UCSF file."""
        subprocess.run(
            [sys.executable, "mkucsf_gpu.py", "--backend", "numpy"],
            cwd=setup_sim_dir,
            capture_output=True
        )

        ucsf_path = os.path.join(setup_sim_dir, "sim_gpu.ucsf")

        # Try to load with nmrglue
        try:
            import nmrglue as ng
            dic, data = ng.sparky.read(ucsf_path)

            assert data.shape == (512, 1024)
            assert data.dtype == np.float32
            assert np.max(data) > 0  # Should have signal

        except ImportError:
            pytest.skip("nmrglue not available")

    def test_reference_vs_gpu_match(self, setup_sim_dir):
        """Test GPU simulation matches reference implementation."""
        # Run both simulations
        subprocess.run(
            [sys.executable, "mkucsf.py"],
            cwd=setup_sim_dir,
            capture_output=True
        )
        subprocess.run(
            [sys.executable, "mkucsf_gpu.py", "--backend", "numpy"],
            cwd=setup_sim_dir,
            capture_output=True
        )

        try:
            import nmrglue as ng

            ref_path = os.path.join(setup_sim_dir, "sim.ucsf")
            gpu_path = os.path.join(setup_sim_dir, "sim_gpu.ucsf")

            dic_ref, data_ref = ng.sparky.read(ref_path)
            dic_gpu, data_gpu = ng.sparky.read(gpu_path)

            # ZNCC should be very high
            a = data_ref.flatten() - np.mean(data_ref)
            b = data_gpu.flatten() - np.mean(data_gpu)
            zncc = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))

            assert zncc > 0.999, f"ZNCC {zncc} too low"

            # Max difference should be small
            max_diff = np.max(np.abs(data_ref - data_gpu))
            assert max_diff < 1e-3, f"Max diff {max_diff} too large"

        except ImportError:
            pytest.skip("nmrglue not available")


class TestBatchProcessing:
    """Integration tests for batch processing."""

    def test_batch_single_file(self, tp5abc_data):
        """Test batch processing with single file."""
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Use clean environment to avoid venv interference
        env = os.environ.copy()
        env.pop('VIRTUAL_ENV', None)

        result = subprocess.run(
            [sys.executable, "genimino_batch.py", "tP5abc.seq"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            env=env
        )

        assert result.returncode == 0, f"Batch failed: {result.stderr}"
