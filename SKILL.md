---
name: audio-polish
description: Automatic sound fixing and mixing for voice-over videos and music beds. Fixes audio that sounds dull/flat in some parts and harsh/sharp in others (adaptive tonal balance plus a harshness-gated resonance tamer), out-of-tune sampled instruments (YIN tuning table plus band-limited retune), uneven voice takes (clean, level, -16 LUFS) and music that is too quiet or fights the voice (speech-relative level, ducking, voice-only 1-4 kHz carve). Masters to -14 LUFS / -1.5 dBTP and prints a QA report. Use when asked to improve, clean up, balance, master or mix audio for a reel, short, explainer, devlog or podcast clip, or when someone says the music is too quiet/loud, muddy, harsh, dull, flat, sharp or out of tune.
---

# Audio polish

Four Python scripts in `scripts/`. They need Python 3.10+, numpy, scipy, soundfile, and ffmpeg on PATH.

| Script | What it does |
|---|---|
| `automix.py` | **The one-command path.** Takes a voice-only video and a music WAV, polishes both, mixes, masters, muxes the picture and runs QA. |
| `audio_polish.py` | `analyze` reports the tone of a file (or `polish`es it), finding and fixing dull and harsh stretches. |
| `voice_chain.py` | Cleans a raw voice take: declip, HPF, denoise, gate, de-mud, presence, de-ess, speechnorm, compression, then two-pass loudnorm to -16 LUFS. |
| `sample_tuning.py` | Measures a sample library's intonation in cents (YIN) and writes `tuning.json` for retuning. |

## Workflow

1. **Voice takes:** run `python scripts/voice_chain.py take raw.wav clean.wav` on each recorded line, then build and render the voice-only video.
2. **Music from samples:** if the music is built from a multisample library, run `sample_tuning.py <samples_dir>` once. In the sampler, shift each note by `semitones*100 - tuning[file]` cents. Use band-limited resampling (`scipy.signal.resample_poly` with a `Fraction(step).limit_denominator(300)` ratio), never `np.interp`. Linear interpolation dulls the highs and aliases upward shifts into harshness.
3. **Mix:** run `python scripts/automix.py voice-only.mp4 music.wav final.mp4`.
   - `--rel -10` sets the music's level under the voice while it speaks, in dB. -10 is clearly audible; -14 and below reads as "barely audible" on phones.
   - `--duck 4` sets how much quieter the music is while the voice talks than in the gaps.
   - `--no-polish` skips steps 1–2 of the polish.
4. **Read the report:**
   - The `harsh` index is 2.5–6.3 kHz and `dull/air` is 6.3–14 kHz, both in dB against a balanced reference, anchored on 250–1000 Hz.
   - The `spread` is the p95–p5 of the harsh index over time. Large spreads mean some parts are sharp and others flat.
   - For music, aim for a spread under ~10 dB and a harsh p95 under about +5 dB.
   - Flags list the stretches still more than 3 dB off. `decode_ok` must be true, and `lufs` should be about -14.
5. Listen if you can. The checks are signal-based only. Never play audio aloud if the user said they're busy or studying.

## How the polish works (audio_polish.py)

- **STFT:** 4096-sample window and 1024 hop at 48 kHz, in 1/3-octave bands from 63 Hz to 16 kHz. Bands are smoothed over about 0.4 s for music and about 2 s for voice, so the voice comparison is per line, not per phoneme.
- **Static correction:** the track's long-term shape moves toward the reference: 75% of the way for music (±8 dB), 50% for voice (±4 dB). The lift above 12.5 kHz is capped at +4 dB so sample hiss doesn't come up.
- **Dynamic correction:** each moment is pulled toward the track's own average, at 0.6 strength (±6 dB). It is time-smoothed, and silent passages are left alone.
- **Resonance tamer:** works on 1.2–10 kHz peaks that stand more than 6 dB above a 1/6-octave envelope. It only acts in moments that are harsher than the track's norm, because harmonics are peaks too. An always-on tamer dulls music.
- **Loudness:** restored to the input's integrated LUFS, with the true peak at or below -1.5 dBTP.

## Lessons learned

- "Flat" and "sharp" can mean pitch **or** tone. Check both. In VSCO 2 CE, the clarinet staccato set is 18–38 cents flat and some harp notes are about 28 cents flat. Most VSCO sets label middle C as C3, but the harp and Upright Nr1 piano use C4. Verify octave labels with YIN before writing notes.
- Fully synthesized (oscillator) scores read as cheap "elevator music". Real recorded instruments are better. Stock music tends to feel generic.
- Peak-normalizing every sample erases velocity-layer loudness. Balance instruments by measured per-section RMS: the melody should sit about 3–5 dB above the accompaniment.
