"""Tests for RNA motif extraction (tools/fraMotif.py)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fraMotif import pairing, getMotif


class TestPairing:
    """Tests for bracket-dot structure parsing."""

    def test_simple_hairpin(self):
        """Test simple hairpin structure."""
        bdstr = "(((....)))"
        pairs = pairing(bdstr)

        # Position 0 pairs with 9, 1 with 8, 2 with 7
        assert pairs[0] == 9
        assert pairs[9] == 0
        assert pairs[1] == 8
        assert pairs[8] == 1
        assert pairs[2] == 7
        assert pairs[7] == 2

        # Loop positions (3-6) should not be in pairs
        assert 3 not in pairs
        assert 4 not in pairs
        assert 5 not in pairs
        assert 6 not in pairs

    def test_nested_structure(self):
        """Test nested stem-loop structure."""
        bdstr = "(((...)))"
        pairs = pairing(bdstr)

        assert pairs[0] == 8
        assert pairs[1] == 7
        assert pairs[2] == 6

    def test_empty_structure(self):
        """Test structure with no base pairs."""
        bdstr = "......"
        pairs = pairing(bdstr)
        assert pairs == {}

    def test_single_pair(self):
        """Test single base pair."""
        bdstr = "(..)"
        pairs = pairing(bdstr)
        assert pairs[0] == 3
        assert pairs[3] == 0


class TestGetMotif:
    """Tests for motif extraction."""

    def test_base_pair_gc(self):
        """Test basePair motif for G-C pair."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        # Position 0 is G paired with C at position 9
        idx, mode = getMotif('basePair', 0, seq, bd)

        assert idx == [0, 9]
        assert 'GC' in mode

    def test_base_pair_unpaired(self):
        """Test basePair for unpaired residue returns False."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        # Position 4 is in the loop (unpaired)
        idx, mode = getMotif('basePair', 4, seq, bd)

        assert idx is False
        assert mode is False

    def test_triplet_valid(self):
        """Test triplet motif for valid stacked base pairs."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        # Position 1 should have valid triplet (positions 0,1,2 all paired)
        idx, mode = getMotif('triplet', 1, seq, bd)

        assert idx is not False
        assert mode is not False
        # Should have 3 base pairs in mode
        assert len([m for m in mode if len(m) == 2 and m[0] in 'GCAU']) >= 3

    def test_triplet_at_edge(self):
        """Test triplet at structure edge returns False."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        # Position 0 can't have a triplet (no -1 position)
        idx, mode = getMotif('triplet', 0, seq, bd)

        # Should fail due to KeyError on idx-1
        assert idx is False or mode is False

    def test_triplet_in_loop(self):
        """Test triplet in loop returns False."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        idx, mode = getMotif('triplet', 5, seq, bd)

        assert idx is False
        assert mode is False

    def test_penta_valid(self, tp5abc_data):
        """Test penta motif with real sequence."""
        seq = tp5abc_data["sequence"]
        bd = tp5abc_data["structure"]

        # Find a position that might have penta motif
        # Need 5 consecutive stacked base pairs
        # This is rare - test that function runs without error
        for i in range(2, len(seq) - 2):
            idx, mode = getMotif('penta', i, seq, bd)
            if idx is not False:
                # If found, should have 5 base pairs
                assert len([m for m in mode if len(m) == 2]) >= 5
                break


class TestMotifBarcode:
    """Tests for barcode generation."""

    def test_barcode_format(self):
        """Test that triplet mode can be joined into barcode."""
        seq = "GGCGAAAGCC"
        bd = "(((....)))"

        idx, mode = getMotif('triplet', 1, seq, bd)

        if mode:
            # First 3 elements are base pairs
            barcode = '-'.join(mode[:3])
            assert barcode.count('-') == 2
            # Each part should be 2 characters (base pair)
            parts = barcode.split('-')
            for part in parts:
                assert len(part) == 2
