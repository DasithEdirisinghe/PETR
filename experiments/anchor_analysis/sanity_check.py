#!/usr/bin/env python
"""One-sample smoke test for the PETR anchor-analysis pipeline."""

from __future__ import print_function

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
command = [sys.executable, str(HERE / 'analyze.py'), '--sanity-only',
           '--output-dir', str(HERE / 'outputs_sanity')]
raise SystemExit(subprocess.call(command))

