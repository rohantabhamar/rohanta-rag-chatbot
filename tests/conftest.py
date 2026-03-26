"""
tests/conftest.py
──────────────────
Shared pytest fixtures and configuration.
"""

import sys
import os

# Ensure the project root is on the path for all tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
