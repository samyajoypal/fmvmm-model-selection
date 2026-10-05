"""Portable, read-only access to the frozen numerical results."""
from pathlib import Path
import csv
import gzip
import json
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TABLES = ROOT / 'build/tables'
def read_json(name):
    path = RESULTS / name
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)
def write_json(name, value):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def write_csv(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    assert rows, name
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES/name).open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=fields);out.writeheader()
        for row in rows:
            out.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
