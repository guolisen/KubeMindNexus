#!/usr/bin/env python
"""
__main__.py for KubeMindNexus package

This module allows the package to be executed directly using:
python -m kubemindnexus [args]

It simply imports and calls the main function from the main module.
"""

import sys
import os

# Add parent directory to sys.path to import main from root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and call the main function from main.py
from main import main

if __name__ == "__main__":
    main()
