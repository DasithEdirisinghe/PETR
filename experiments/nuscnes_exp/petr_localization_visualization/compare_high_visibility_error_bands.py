#!/usr/bin/env python3
"""Make PETR error-band videos for TPs with 80–100% visible assigned GTs.

Uses the saved PETR result JSON. By default, selects from all validation
scenes and writes alongside, rather than over, the unfiltered videos.
"""
import sys
from pathlib import Path

from compare_error_bands import main


def has_option(arguments, option):
    return any(value == option or value.startswith(option + '=')
               for value in arguments)


if __name__ == '__main__':
    arguments = list(sys.argv[1:])
    if not any(has_option(arguments, option)
               for option in ('--all-val', '--scenes', '--scene-count')):
        arguments.insert(0, '--all-val')
    if not has_option(arguments, '--output-dir'):
        arguments[:0] = [
            '--output-dir', str(Path(__file__).resolve().parent /
                                'outputs/high_visibility_error_band_comparisons')]
    if has_option(arguments, '--visibility-tokens'):
        raise SystemExit('This entry point fixes --visibility-tokens to 4; '
                         'use compare_error_bands.py for another filter.')
    main(arguments + ['--visibility-tokens', '4'], description=__doc__,
         hide_visibility_option=True)
