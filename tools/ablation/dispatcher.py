#!/usr/bin/python3 -I
"""Dispatch the fixed ablation extension; preserve the installed main CLI."""
import os, sys
BASE = '/srv/digit-ae/admin/ablation_v1/'
args = sys.argv[1:]
ablation = bool(args and args[0] == 'ablation') or any(
    x == '--action=ablation' or x == '--action' and i + 1 < len(args) and args[i + 1] == 'ablation'
    for i, x in enumerate(args))
if args in (['--help'], ['-h']):
    print('Supplementary: digit-ae ablation PA sage; inspect with --action ablation', flush=True)
os.execv('/usr/bin/python3', ['/usr/bin/python3', '-I', '-B', BASE + ('cli.py' if ablation else 'legacy_cli.py'), *args])
