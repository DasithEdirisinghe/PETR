#!/usr/bin/env python
"""Compatibility entry point for the keyframe-selection experiment."""

import runpy
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[2] / 'experiments' /
          'keyframe_selection' / 'select_petr_keyframe.py')
runpy.run_path(str(SCRIPT), run_name='__main__')
