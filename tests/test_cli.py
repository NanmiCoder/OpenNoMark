"""Tests for CLI interface."""

import os
import subprocess
import sys
import pytest


class TestCLI:
    """Test the opennomark CLI tool."""

    def run_cli(self, *args, timeout=300):
        """Helper to run CLI and capture output."""
        result = subprocess.run(
            [sys.executable, "-m", "opennomark.cli", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        return result

    def test_help(self):
        result = self.run_cli("--help")
        assert result.returncode == 0
        assert "watermark" in result.stdout.lower()
        assert "inputs" in result.stdout

    def test_no_args_fails(self):
        result = self.run_cli()
        assert result.returncode != 0

    def test_nonexistent_input(self):
        result = self.run_cli("/nonexistent/path.png", "-o", "/tmp/test_out")
        assert result.returncode != 0
        assert "No valid images" in result.stderr

    def test_single_file(self, sample_image, output_dir):
        result = self.run_cli(sample_image, "-o", output_dir)
        assert result.returncode == 0
        assert "Done!" in result.stdout
        assert "1 image" in result.stdout

    def test_directory_input(self, sample_images_dir, output_dir):
        result = self.run_cli(sample_images_dir, "-o", output_dir)
        assert result.returncode == 0
        assert "Found 3 image" in result.stdout
        assert "Done!" in result.stdout

    def test_multiple_inputs(self, sample_images_dir, output_dir):
        # Pass the same dir twice to test multiple args
        result = self.run_cli(sample_images_dir, sample_images_dir, "-o", output_dir)
        assert result.returncode == 0
        assert "Done!" in result.stdout

    def test_debug_flag(self, sample_image, tmp_path):
        out = str(tmp_path / "debug_output")
        result = self.run_cli(sample_image, "-o", out, "--debug")
        assert result.returncode == 0

    def test_real_gemini_dir(self, output_dir):
        gemini_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gemini_images")
        if not os.path.isdir(gemini_dir):
            pytest.skip("gemini_images not available")
        result = self.run_cli(gemini_dir, "-o", output_dir)
        assert result.returncode == 0
        assert "cleaned" in result.stdout
