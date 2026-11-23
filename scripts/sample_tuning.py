"""Measure the intonation of a pitched sample library and write a per-file cents table.

  python sample_tuning.py <samples_dir> [--out tuning.json] [--glob "**/*.wav"]

File names must contain a note label such as _A#3_ or _C4. (the octave convention does
not matter: offsets are folded to +/-600 cents). Each file is analysed with YIN over its
steady part (attack skipped); the median offset in cents is stored when the track is
stable. A sampler then retunes with:  cents_to_shift = semitones*100 - table[file].

Prints per-folder medians and every file more than 15 cents out.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000
NOTE = re.compile(r'[_\-]([A-G])(#?)(\d)[_.\-]')


def label_midi(name):
    m = NOTE.search(name)
    return None if not m else 12 * (int(m[3]) + 1) + 'C D EF G A B'.index(m[1]) + bool(m[2])


def yin_track(x, fmin=50, fmax=2200, W=2048, hop=1024):
    taus = np.arange(int(SR / fmax), int(SR / fmin))
    out = []
    for st in range(0, len(x) - W - taus[-1], hop):
        fr = x[st:st + W + taus[-1]]
        if np.sqrt((fr[:W] ** 2).mean()) < 0.02:
            continue
        d = np.array([np.sum((fr[:W] - fr[t:t + W]) ** 2) for t in taus])
        cm = d * np.arange(1, len(d) + 1) / (np.cumsum(d) + 1e-12)
        idx = np.where(cm < 0.12)[0]
        if not len(idx):
            continue
        i = idx[0]
        while i + 1 < len(cm) and cm[i + 1] < cm[i]:
            i += 1
        if 0 < i < len(cm) - 1:
            a, b, c = cm[i - 1], cm[i], cm[i + 1]
            i = i + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        out.append(SR / (taus[0] + i))
    return np.array(out)


def cents(path):
    midi = label_midi(path.name)
    if midi is None:
        return None
    a, sr = sf.read(path, always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        a = signal.resample_poly(a, SR, sr)
    a = a / (np.abs(a).max() + 1e-9)
    on = int(np.argmax(np.abs(a) > 0.02))
    a = a[on + int(0.08 * SR): on + int(1.6 * SR)]
    f = yin_track(a)
    if len(f) < 3:
        return None
    c = 1200 * np.log2(f / (440 * 2 ** ((midi - 69) / 12)))
    c = c - 1200 * np.round(c / 1200)
    c = c[np.abs(c - np.median(c)) < 60]
    if len(c) < 3:
        return None
    med = float(np.median(c))
    return round(med, 1) if abs(med) < 80 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir'); p.add_argument('--out', default='tuning.json'); p.add_argument('--glob', default='**/*.wav')
    a = p.parse_args()
    root = Path(a.dir)
    res = {}
    for f in sorted(root.glob(a.glob)):
        c = cents(f)
        if c is not None:
            res[f.relative_to(root).as_posix()] = c
    Path(a.out).write_text(json.dumps(res, indent=1))
    by = {}
    for k, v in res.items():
        by.setdefault(k.rsplit('/', 1)[0] if '/' in k else '.', []).append(v)
    for k, v in sorted(by.items()):
        v = np.array(v)
        print(f'{k}: n={len(v)} median {np.median(v):+.1f}c, worst {v[np.argmax(abs(v))]:+.1f}c')
    for k, v in res.items():
        if abs(v) > 15:
            print(f'  out of tune {v:+.1f}c  {k}')


if __name__ == '__main__':
    main()
