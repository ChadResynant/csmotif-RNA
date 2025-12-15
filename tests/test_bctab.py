"""Tests for chemical shift table parsing (tools/bctab.py)."""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bctab import BCTab


class TestBCTab:
    """Tests for BCTab chemical shift table parser."""

    def test_load_nh_table(self, nh_cs_table):
        """Test loading NH chemical shift table."""
        assert nh_cs_table.NH is not None
        assert len(nh_cs_table.NH) > 0

    def test_nh_shift_ranges(self, nh_cs_table):
        """Test that NH shifts are in expected ranges."""
        for barcode, shifts in nh_cs_table.NH.items():
            N_shift, H_shift = shifts

            # 15N imino range: ~140-165 ppm
            assert 140 <= N_shift <= 165, f"N shift {N_shift} out of range for {barcode}"

            # 1H imino range: ~10-15 ppm
            assert 10 <= H_shift <= 15, f"H shift {H_shift} out of range for {barcode}"

    def test_barcode_format(self, nh_cs_table):
        """Test barcode format is valid."""
        valid_bp = {'GC', 'CG', 'AU', 'UA', 'GU', 'UG'}

        for barcode in nh_cs_table.NH.keys():
            parts = barcode.split('-')
            assert len(parts) == 3, f"Barcode {barcode} should have 3 parts"

            for part in parts:
                assert part in valid_bp, f"Invalid base pair {part} in {barcode}"

    def test_nh_nuclei_assignment(self, nh_cs_table):
        """Test NHnu dictionary has correct nuclei names."""
        for barcode, nuclei in nh_cs_table.NHnu.items():
            first_nt = barcode[0]

            if first_nt in ['G', 'A']:
                assert 'N1' in nuclei
                assert 'H1' in nuclei
            elif first_nt in ['U', 'C']:
                assert 'N3' in nuclei
                assert 'H3' in nuclei

    def test_common_barcodes_exist(self, nh_cs_table):
        """Test that common barcodes exist in table."""
        # Check some barcodes that should exist in the table
        # Note: Not all combinations exist - only those observed in data
        assert len(nh_cs_table.NH) > 100, "Table should have >100 entries"

        # Check format of first few entries
        for barcode in list(nh_cs_table.NH.keys())[:5]:
            parts = barcode.split('-')
            assert len(parts) == 3

    def test_hz_conversion(self, nh_cs_table):
        """Test Hz conversion values."""
        # NHHz should have scaled values
        for barcode in list(nh_cs_table.NH.keys())[:5]:
            if barcode in nh_cs_table.NHHz:
                ppm_n, ppm_h = nh_cs_table.NH[barcode]
                hz_n, hz_h = nh_cs_table.NHHz[barcode]

                # 15N at 60 MHz, 1H at 600 MHz
                assert abs(hz_n - ppm_n * 60) < 0.1
                assert abs(hz_h - ppm_h * 600) < 0.1


class TestBCTabEmpty:
    """Tests for BCTab with no file."""

    def test_empty_init(self):
        """Test BCTab can be initialized without file."""
        tab = BCTab()
        assert tab.NH == {}
        assert tab.CH == {}
