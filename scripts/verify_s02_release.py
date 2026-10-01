#!/usr/bin/env python3
"""Check published source identities and recompute the reported comparison."""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
results = root / 'results/2026-10-01'
manifest = json.loads((results / 's02_source_sha256.json').read_text())
actual = {str(p.relative_to(root)) for p in (root / 'runtime/s02').rglob('*.py')}
assert actual == set(manifest)
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
data = json.loads((results / 'fixed_int8.json').read_text())
assert data['short_order'] == ['A1', 'B1', 'B2', 'A2', 'A3', 'B3']
assert len(data['short_pairs']) == 3 and len(data['long_pairs']) == 1
def savings(pair):
    assert 0 < pair['b_s'] < pair['a_s']
    return 100 * (1 - pair['b_s'] / pair['a_s'])
short = statistics.median(map(savings, data['short_pairs']))
long = savings(data['long_pairs'][0])
assert abs(short - 10.754111) < 0.000001
assert abs(long - 6.655705) < 0.000001
print(f'S02: {len(manifest)} source hashes; short paired median {short:.6f}%; long first pair {long:.6f}%.')
