"""One command from voice-only render + music bed to a finished, QA'd social video.

  python automix.py <voice-only.mp4> <music.wav> <out.mp4> [--rel -10] [--duck 4] [--no-polish]

Steps
 1. Polish the voice (audio_polish, profile voice): line-to-line tone made consistent, harsh/sibilant moments tamed.
 2. Polish the music (profile music): long-term tone moved toward a balanced reference, dull and harsh stretches evened out.
 3. Mix: the music sits `--rel` dB under the voice while it speaks (RMS over speech blocks), ducks `--duck` dB
    with a 60 ms attack / 500 ms release, and gets a 1-4 kHz dip only while the voice is talking.
 4. Master to -14 LUFS / -1.5 dBTP, mux with the original picture (video stream copied).
 5. QA: decode check, loudness, music-vs-voice levels, tonal report of the final mix.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_polish as ap

SR = ap.SR


def speech_mask(v, block=0.05):
    n = int(SR * block)
    m = len(v) // n
    rms = np.sqrt((v[:m * n].mean(axis=1).reshape(m, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-9)
    act = db > (np.percentile(db, 95) - 24)
    mask = np.concatenate([np.repeat(act, n), np.zeros(len(v) - m * n, bool)])
    return mask


def follower(gate, attack=0.06, release=0.5, sub=48):
    gs = gate[::sub].astype(float)
    a1, r1 = np.exp(-sub / (SR * attack)), np.exp(-sub / (SR * release))
    ys, y = np.empty_like(gs), 0.0
    for i, x in enumerate(gs):
        c = a1 if x > y else r1
        y = c * y + (1 - c) * x
        ys[i] = y
    return np.interp(np.arange(len(gate)), np.arange(len(gs)) * sub, ys)


def mix(voice, music, rel_db=-10.0, duck_db=4.0):
    music = np.vstack([music, np.zeros((max(0, len(voice) - len(music)), 2))])[:len(voice)]
    sp = speech_mask(voice)
    g = follower(sp)
    # dynamic carve: 1-4 kHz of the music dips ~4 dB only while the voice talks
    sos = signal.butter(2, [1000, 4000], fs=SR, btype='bandpass', output='sos')
    mid = signal.sosfiltfilt(sos, music, axis=0)
    music = music - mid * (1 - 10 ** (-4 / 20)) * g[:, None]
    music = music * (10 ** (-duck_db * g / 20))[:, None]
    rv = np.sqrt((voice[sp] ** 2).mean())
    rm = np.sqrt((music[sp] ** 2).mean())
    music *= 10 ** (rel_db / 20) * rv / rm
    stats = {'music_vs_voice_speech_db': round(20 * np.log10(np.sqrt((music[sp] ** 2).mean()) / rv), 1),
             'music_vs_voice_gaps_db': round(20 * np.log10(np.sqrt((music[~sp] ** 2).mean()) / rv), 1)}
    return voice + music, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video'); p.add_argument('music'); p.add_argument('out')
    p.add_argument('--rel', type=float, default=-10.0)
    p.add_argument('--duck', type=float, default=4.0)
    p.add_argument('--no-polish', action='store_true')
    a = p.parse_args()
    voice = ap.load(a.video)
    music = ap.load(a.music)
    if not a.no_polish:
        print('voice:'); ap.report(voice, 'voice', 'before')
        voice = ap.polish(voice, 'voice')
        ap.report(voice, 'voice', 'after')
        print('music:'); ap.report(music, 'music', 'before')
        music = ap.polish(music, 'music')
        ap.report(music, 'music', 'after')
    x, stats = mix(voice, music, a.rel, a.duck)
    tmp = Path(a.out).with_suffix('.premix.wav')
    sf.write(tmp, x, SR, subtype='FLOAT')
    i, _ = ap.lufs(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.video, '-i', str(tmp), '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                    '-af', f'volume={-14 - i:.3f}dB,alimiter=limit=0.84:attack=3:release=50:level=false',
                    '-c:a', 'aac', '-b:a', '256k', '-ar', str(SR), '-movflags', '+faststart', a.out], check=True)
    tmp.unlink()
    ok = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', a.out, '-f', 'null', '-'], capture_output=True).returncode == 0
    I, TP = ap.lufs(a.out)
    print('final mix:'); _, flags = ap.report(ap.load(a.out), 'mix', 'final')
    print(json.dumps({'out': a.out, 'decode_ok': ok, 'lufs': I, 'true_peak': TP, **stats}))


if __name__ == '__main__':
    main()
