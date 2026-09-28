#!/usr/bin/env python3
"""Recompute archived Singularity results; no CUDA or remote connections."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(media=False):
    folder = ROOT / 'results/2026-09-28-singularity'
    data = json.loads((folder / 'benchmark.json').read_text())
    medians = {}
    for name, group in data['groups'].items():
        rows = group['all_iterations']
        assert [r['run'] for r in rows] == [0, 1, 2, 3]
        for row in rows:
            assert row['output_shape'] == [56, 672, 1152, 3]
            assert row['sampling_total_s'] == row['first_sampling_s'] + row['second_sampling_s']
            assert row['decoding_total_s'] == row['first_decode_s'] + row['second_decode_s']
        medians[name] = {k: statistics.median(r[k] for r in rows[1:]) for k in group['warm_median']}
        assert medians[name] == group['warm_median']
        assert group['warm_total_range'] == [min(r['total_s_including_evidence'] for r in rows[1:]), max(r['total_s_including_evidence'] for r in rows[1:])]
        assert group['exit']['exit'] == 0
    total = 'total_s_including_evidence'
    assert data['fair_single_baseline'] == 'native_single'
    single, old, dual = (medians[n][total] for n in ('native_single', 'original_dual', 'optimized_dual'))
    assert single < medians['cached_single'][total]
    assert data['fair_dual_speedup'] == single / dual
    assert data['improvement_over_original_dual'] == old / dual
    assert abs(data['elapsed_reduction_vs_original_dual_percent'] - (1 - dual / old) * 100) < 1e-10

    sigmas = json.loads((folder / 'sigmas.json').read_text())
    assert len(sigmas['first']) - 1 == 2 and len(sigmas['second']) - 1 == 10
    assert len(sigmas['zero']) == 1
    adaln = json.loads((folder / 'adaln.json').read_text())
    assert adaln['ported'] == 50 and adaln['stripped'] == adaln['unportable'] == 0
    assert adaln['effective_mode'] == 'port'

    media_data = json.loads((folder / 'media_checks.json').read_text())
    checks = media_data['checks']
    assert media_data['count'] == len(checks) == 54
    assert len({(r['run'], r['stage'], r['iteration']) for r in checks}) == 54
    benchmark_runs = {g['run'] for g in data['groups'].values()}
    selected = [r for r in checks if r['run'] in benchmark_runs]
    assert {(r['run'], r['stage'], r['iteration']) for r in selected} == {(run, stage, i) for run in benchmark_runs for stage in ('first', 'second') for i in range(4)}
    for row in checks:
        assert row['full_decode_exit'] == 0 and row['latent_exact'] and row['float_pixels_exact']
        assert row['pixels']['finite'] and all(row['latents']['finite'])
        assert len(row['latents']['comparison']) == 2
        assert all(c['exact'] and c['max_abs'] == c['relative_l2'] == 0 for c in row['latents']['comparison'])
        dims = (768, 448) if row['stage'] == 'first' else (1152, 672)
        assert row['pixels']['shape'] == [56, dims[1], dims[0], 3]
        video = next(s for s in row['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height'], video['nb_frames'], video['avg_frame_rate']) == (*dims, '56', '24/1')
        if row['run'] in benchmark_runs:
            assert row['mp4_sha256'] == data['mp4_sha256'][row['stage']]
    for stage in ('first', 'second'):
        assert len({r['pixels']['sha256_float_pixels'] for r in selected if r['stage'] == stage}) == 1
    ref = json.loads((folder / 'reference.json').read_text())
    assert hashlib.sha256((ROOT / ref['reference_video']).read_bytes()).hexdigest() == ref['reference_video_sha256']
    assert len(ref['iterations']) == 1
    samples = [('singularity-sailboat-1152x672-56f.mp4', 'r020_singularity_prefetch_dual', 3),
               ('singularity-reference-1152x672-56f.mp4', ref['run'], 0)]
    for name, run, iteration in samples:
        clip = ROOT / 'files' / name
        evidence = next(r for r in checks if (r['run'], r['stage'], r['iteration']) == (run, 'second', iteration))
        assert hashlib.sha256(clip.read_bytes()).hexdigest() == evidence['mp4_sha256']
        if media:
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(clip)], text=True))
            v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            assert (v['width'], v['height'], v['nb_frames'], v['avg_frame_rate']) == (1152, 672, '56', '24/1')
            assert (a['codec_name'], a['sample_rate'], a['channels']) == ('aac', '32000', 2)
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(clip), '-f', 'null', '-'], check=True)
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, digest in provenance['public_source_sha256'].items():
        assert hashlib.sha256((ROOT / 'runtime/singularity' / name).read_bytes()).hexdigest() == digest, name
    print('Singularity: %.6fx total; %.6fx sampling; %.4f%% less elapsed time vs old dual; 54 archived checks, 32 benchmark files, 2 sample hashes verified%s.' %
          (single / dual, medians['native_single']['sampling_total_s'] / medians['optimized_dual']['sampling_total_s'],
           (1 - dual / old) * 100, '; samples fully decoded' if media else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', action='store_true')
    verify(parser.parse_args().media)
