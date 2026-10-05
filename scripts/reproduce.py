#!/usr/bin/env python3
"""One command regenerates paper figures and numerical tables; no fitting."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true',help='verify frozen input hashes without generating outputs')
    args=p.parse_args()
    manifest=json.loads((ROOT/'results/manifest.json').read_text())
    for name,item in manifest['files'].items():
        data=(ROOT/'results'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==item['sha256'],name
        assert hashlib.sha256(gzip.decompress(data)).hexdigest()==item['json_sha256'],name
    print(f"Verified {len(manifest['files'])} numerical input files.",flush=True)
    if args.verify_only:return
    for d in ['figures','tables']:(ROOT/'build'/d).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'build/.matplotlib'))
    for name in ['real_results.py','baseline_figures.py','comparison_figures.py','paper_tables.py']:
        subprocess.run([sys.executable,str(ROOT/'analysis'/name)],cwd=ROOT,env=env,check=True)
    print('Complete. Figures: build/figures; CSV/JSON tables and checks: build/tables.',flush=True)
if __name__=='__main__':main()
