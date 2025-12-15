"""Pytest configuration and fixtures for csmotif-RNA tests."""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_sequence():
    """Simple RNA hairpin sequence for testing."""
    return "GGCGAAAGCC"


@pytest.fixture
def sample_structure():
    """Bracket-dot structure matching sample_sequence."""
    return "(((....)))"


@pytest.fixture
def tp5abc_sequence():
    """Full tP5abc sequence from test file."""
    return "GGCAGUACCAAGUCGCGAAAGCGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGCC"


@pytest.fixture
def tp5abc_structure():
    """Full tP5abc structure from test file."""
    return "(((((((((..(((((....))))).(((((....)))))..))).....)))))))".rstrip(")")
    # Note: actual structure has matching parens


@pytest.fixture
def tp5abc_data():
    """Load actual tP5abc.seq file."""
    seq_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tP5abc.seq"
    )
    with open(seq_file) as f:
        lines = f.readlines()
    return {
        "sequence": lines[0].strip(),
        "structure": lines[1].strip()
    }


@pytest.fixture
def nh_cs_table():
    """Load NH chemical shift table."""
    from tools.bctab import BCTab
    cs_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "NH.cs"
    )
    return BCTab(cs_file)


@pytest.fixture
def gpu_backend():
    """Initialize GPU backend for tests, return backend name."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sim.imino"
    ))
    from mkucsf_gpu import init_gpu_backend, GPU_BACKEND
    init_gpu_backend()
    # Re-import to get updated value
    from mkucsf_gpu import GPU_BACKEND
    return GPU_BACKEND
