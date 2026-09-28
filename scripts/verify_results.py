#!/usr/bin/env python3
"""Recompute the published medians and verify their evidence without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(media=False):
    folder = ROOT / 'results/2026-09-28'
    summary = json.loads((folder / 'summary.json').read_text())
    recomputed = {}
    for series in ('single', 'dual_initial', 'dual_optimized'):
        rows = json.loads((folder / (series + '.json')).read_text())
        assert [r['run'] for r in rows] == [0, 1, 2, 3]
        assert all((r['width'], r['height'], r['frames']) == (832, 480, 56) for r in rows)
        recomputed[series] = {k: statistics.median(r[k] for r in rows[1:])
                              for k in ('conditioning_s', 'sampling_s', 'decode_s', 'save_s', 'total_s')}
        assert recomputed[series] == summary['medians_seconds'][series]
    ratios = {k: recomputed['single'][k] / recomputed['dual_optimized'][k] for k in recomputed['single']}
    assert ratios == summary['speedup_single_over_optimized']
    checks = json.loads((folder / 'media_checks.json').read_text())
    assert len(checks) == 12
    assert {(r['series'], r['run']) for r in checks} == {(s, r) for s in recomputed for r in range(4)}
    assert all(r['full_decode_exit'] == 0 for r in checks)
    hashes = {r['sha256'] for r in checks}
    assert len(hashes) == 1
    clip = ROOT / 'files/sailboat-832x480-56f.mp4'
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == next(iter(hashes))
    parity = json.loads((folder / 'vae_parity.json').read_text())
    assert parity == {'exact': True, 'max_abs': 0.0, 'shape': [56, 480, 832, 3]}
    for shape in json.loads((folder / 'benchmark.json').read_text()):
        assert len(shape['variants']) == 6
        for variant in shape['variants']:
            assert variant['numerical_gate'] and len(variant['seconds']) == 3
            assert all(e['finite'] and e['relative_max'] == e['relative_l2'] == 0 for e in variant['errors'])
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, digest in provenance['public_source_sha256'].items():
        assert hashlib.sha256((ROOT / 'runtime' / name).read_bytes()).hexdigest() == digest, name
    if media:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(clip)], text=True))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        assert (video['width'], video['height'], video['nb_frames'], video['avg_frame_rate']) == (832, 480, '56', '24/1')
        assert (audio['codec_name'], audio['sample_rate'], audio['channels']) == ('aac', '32000', 2)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(clip), '-f', 'null', '-'], check=True)
    print('Verified: total %.6fx; sampling %.6fx; 12 matching recorded media hashes; sample hash checked%s.' %
          (ratios['total_s'], ratios['sampling_s'], '; sample probed and fully decoded' if media else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--media', action='store_true', help='also run ffprobe and full ffmpeg decode on the sample')
    verify(p.parse_args().media)
