"""Automatic tonal polish for music beds, voice takes and final mixes.

Fixes the two complaints "some parts sound flat (dull) and some sharp (harsh)":

1. Adaptive tonal balance: every ~0.4 s, the 1/3-octave spectrum is pulled toward a
   target shape. The target is a blend of the track's own long-term average and a
   reference curve for the profile. Corrections are smoothed in time and frequency
   and clamped, and quiet passages are left alone so noise is never boosted.
2. Resonance tamer (soothe-style): in the presence band, narrow peaks that stick out
   of the local spectral envelope are pressed down dynamically. Harsh string and reed
   resonances and voice sibilance spikes are reduced without dulling the rest.
3. Loudness is restored to the input's integrated LUFS (true peak at or below -1.5 dBTP).

  python audio_polish.py analyze <file.wav|mp4> [--profile music|voice|mix]
  python audio_polish.py polish  <in.wav|mp4> <out.wav> [--profile music|voice|mix] [--strength 0.6]

The analyzer prints a harsh/dull index per stretch (dB versus the reference curve)
and flags stretches beyond +/-3 dB. Polish prints the same numbers before and after.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

SR = 48000
NFFT, HOP = 4096, 1024
CENTERS = 1000 * 2 ** (np.arange(-14, 15) / 3)  # 1/3 octave, 63 Hz .. 16 kHz

# reference 1/3-octave band-energy shapes (dB, relative), from long-term spectra of well
# balanced material: orchestral/acoustic music and close-mic speech
def reference(profile):
    f = CENTERS
    if profile == 'voice':
        r = np.where(f < 150, -6 * np.log2(150 / f), 0.0)
        r += np.where(f > 600, -4.0 * np.log2(f / 600), 0.0)
        r += np.where((f > 2000) & (f < 5000), 2.0, 0.0)  # intelligibility presence
        r += np.where(f > 8000, -3.0 * np.log2(f / 8000), 0.0)
    else:  # music / mix
        r = np.where(f < 60, -6 * np.log2(60 / f), 0.0)
        r += np.where(f > 400, -3.0 * np.log2(f / 400), 0.0)
        r += np.where(f > 2500, -2.0 * np.log2(f / 2500), 0.0)
    return r - r[(f >= 250) & (f <= 1000)].mean()


def load(path):
    p = str(path)
    if p.lower().endswith('.wav'):
        x, sr = sf.read(p, always_2d=True)
        if sr != SR:
            x = signal.resample_poly(x, SR, sr, axis=0)
        return x
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-map', 'a:0', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def lufs(path_or_x):
    if isinstance(path_or_x, np.ndarray):
        tmp = Path(__file__).with_name('.polish-measure.wav')
        sf.write(tmp, path_or_x, SR, subtype='FLOAT')
        src = str(tmp)
    else:
        src, tmp = str(path_or_x), None
    o = subprocess.run(['ffmpeg', '-hide_banner', '-i', src, '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True).stderr
    if tmp:
        tmp.unlink()
    j = json.loads(o[o.rfind('{'):o.rfind('}') + 1])
    return float(j['input_i']), float(j['input_tp'])


def stft(x):
    f, t, Z = signal.stft(x.T, SR, window='hann', nperseg=NFFT, noverlap=NFFT - HOP, boundary='even')
    return f, t, Z  # Z: (ch, bins, frames)


def band_matrix(freqs):
    """(bands, bins) weights for 1/3-octave band energies."""
    lo, hi = CENTERS * 2 ** (-1 / 6), CENTERS * 2 ** (1 / 6)
    return np.array([((freqs >= a) & (freqs < b)).astype(float) for a, b in zip(lo, hi)])


def band_db(Z, freqs, smooth_frames=19):
    P = (np.abs(Z) ** 2).mean(axis=0)  # power, mono sum
    B = band_matrix(freqs) @ P  # (bands, frames)
    B = uniform_filter1d(B, smooth_frames, axis=1, mode='nearest')  # ~0.4 s
    return 10 * np.log10(B + 1e-14)


def indices(bdb, ref):
    """harsh = 2.5-6 kHz, dull = 6-14 kHz, mud = 160-400 Hz; dB vs reference, each anchored on 250-1k."""
    anchor = bdb[(CENTERS >= 250) & (CENTERS <= 1000)].mean(axis=0)
    d = bdb - anchor[None] - ref[:, None]
    pick = lambda a, b: d[(CENTERS >= a) & (CENTERS <= b)].mean(axis=0)
    return {'harsh': pick(2500, 6300), 'dull': pick(6300, 14000), 'mud': pick(160, 400)}, d


def active_mask(bdb, below=40):
    tot = 10 * np.log10((10 ** (bdb / 10)).sum(axis=0) + 1e-14)
    return tot > np.percentile(tot, 95) - below


def smooth_for(profile):
    # voice: compare whole lines (~2 s), not phonemes; music/mix: ~0.4 s
    return 95 if profile == 'voice' else 19


def report(x, profile, label=''):
    f, t, Z = stft(x)
    bdb = band_db(Z, f, smooth_for(profile))
    idx, _ = indices(bdb, reference(profile))
    act = active_mask(bdb)
    out = {k: (float(np.median(v[act])), float(np.percentile(v[act], 95)), float(np.percentile(v[act], 5))) for k, v in idx.items()}
    print(f'{label:8s} harsh {out["harsh"][0]:+5.1f} dB (p95 {out["harsh"][1]:+5.1f})   '
          f'dull/air {out["dull"][0]:+5.1f} dB (p5 {out["dull"][2]:+5.1f})   mud {out["mud"][0]:+5.1f} dB   '
          f'| spread of harsh index p95-p5 {out["harsh"][1]-out["harsh"][2]:.1f} dB')
    # flagged stretches
    flags = []
    for name, sign in (('harsh', 1), ('dull', -1)):
        bad = act & (sign * (idx[name] - np.median(idx[name][act])) > 3)
        if name == 'dull':
            bad = act & (idx['dull'] < np.median(idx['dull'][act]) - 3)
        runs = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
        for a, b in zip(runs[::2], runs[1::2]):
            if t[min(b, len(t) - 1)] - t[a] >= 0.3:
                flags.append((name, round(float(t[a]), 2), round(float(t[min(b, len(t) - 1)]), 2)))
    return out, flags


def polish(x, profile='music', strength=0.6, max_db=6.0, tame_db=6.0):
    f, t, Z = stft(x)
    ref = reference(profile)
    bdb = band_db(Z, f, smooth_for(profile))
    _, dev = indices(bdb, ref)
    act = active_mask(bdb)
    # 1) static: move the track's long-term shape most of the way to the reference
    own = np.median(dev[:, act], axis=1)
    s_amt, s_max = (0.5, 4.0) if profile == 'voice' else (0.75, 8.0)
    static = np.clip(-own * s_amt, -s_max, s_max)
    static[CENTERS < 80] *= 0.3  # leave the deep low end mostly alone
    static[CENTERS > 12500] = np.minimum(static[CENTERS > 12500], 4.0)  # don't lift sample hiss
    # 2) dynamic: pull each moment toward the track's own average, so no stretch is duller/harsher than the rest
    dyn = np.clip(-(dev - own[:, None]) * strength, -max_db, max_db)
    dyn[CENTERS < 80] *= 0.3
    dyn *= act.astype(float)[None]
    corr = static[:, None] + dyn
    corr = gaussian_filter1d(corr, 6 if profile != 'voice' else 20, axis=1)  # time smoothing
    corr = gaussian_filter1d(corr, 0.8, axis=0)  # across neighbouring bands
    # bands -> bins (log-frequency interpolation)
    lf = np.log2(np.maximum(f, 20))
    G = np.array([np.interp(lf, np.log2(CENTERS), corr[:, k]) for k in range(corr.shape[1])]).T  # (bins, frames)

    # resonance tamer, 1.2-10 kHz
    mag = np.abs(Z).mean(axis=0) + 1e-12
    mdb = 20 * np.log10(mag)
    lg = np.linspace(np.log2(40), np.log2(20000), 1024)
    on_grid = np.array([np.interp(lg, lf, mdb[:, k]) for k in range(mdb.shape[1])]).T
    env = gaussian_filter1d(on_grid, 10, axis=0)  # ~1/6 octave envelope
    env_bins = np.array([np.interp(lf, lg, env[:, k]) for k in range(env.shape[1])]).T
    excess = mdb - env_bins - 6.0
    band = ((f > 1200) & (f < 10000)).astype(float)[:, None]
    # only where this moment is harsher than the track's norm (harmonics are peaks too, so never always-on)
    hz = (CENTERS >= 2500) & (CENTERS <= 6300)
    hard = (dev[hz] - own[hz, None]).mean(axis=0)
    w = np.clip(hard / 4.0, 0, 1)[None]
    cut = -np.clip(excess * 0.6, 0, tame_db) * band * w
    cut = uniform_filter1d(cut, 3, axis=1)
    G = G + cut
    Zp = Z * (10 ** (G / 20))[None]
    _, y = signal.istft(Zp, SR, window='hann', nperseg=NFFT, noverlap=NFFT - HOP, boundary=True)
    y = y.T[:len(x)]
    if len(y) < len(x):
        y = np.vstack([y, np.zeros((len(x) - len(y), 2))])
    return y


def restore_loudness(y, target_i, out):
    tmp = Path(out).with_suffix('.pre.wav')
    sf.write(tmp, y, SR, subtype='FLOAT')
    i, _ = lufs(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(tmp), '-af',
                    f'volume={target_i - i:.3f}dB,alimiter=limit=0.84:attack=3:release=50:level=false',
                    '-ar', str(SR), '-c:a', 'pcm_s24le', str(out)], check=True)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['analyze', 'polish'])
    ap.add_argument('src')
    ap.add_argument('dst', nargs='?')
    ap.add_argument('--profile', default='music', choices=['music', 'voice', 'mix'])
    ap.add_argument('--strength', type=float, default=0.6)
    a = ap.parse_args()
    x = load(a.src)
    before, flags = report(x, a.profile, 'before' if a.mode == 'polish' else 'track')
    for fl in flags:
        print('  flag', *fl)
    if a.mode == 'polish':
        i0, _ = lufs(a.src) if a.src.lower().endswith('.wav') else lufs(x)
        y = polish(x, a.profile, a.strength)
        restore_loudness(y, i0, a.dst)
        after, flags2 = report(load(a.dst), a.profile, 'after')
        for fl in flags2:
            print('  flag', *fl)
        i1, tp1 = lufs(a.dst)
        print(f'loudness {i0:.2f} -> {i1:.2f} LUFS, TP {tp1:.2f} dBTP')


if __name__ == '__main__':
    main()
