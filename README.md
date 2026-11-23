# audio-polish

An agent skill (Claude Code / Codex) and a set of standalone Python scripts that automatically fix and mix audio for voice-over videos.

- **automix.py:** one command turns a voice-only video and a music bed into a mastered video (-14 LUFS / -1.5 dBTP) with a QA report.
- **audio_polish.py:** finds and fixes dull and harsh stretches. It uses an adaptive 1/3-octave tonal balance and a harshness-gated resonance tamer.
- **voice_chain.py:** cleans and levels raw voice takes to -16 LUFS.
- **sample_tuning.py:** measures the intonation of a sample library in cents (YIN), so a sampler can retune every note to A=440.

```
pip install numpy scipy soundfile   # plus ffmpeg on PATH
python scripts/automix.py voice-only.mp4 music.wav final.mp4 --rel -10 --duck 4
python scripts/audio_polish.py analyze song.wav
python scripts/audio_polish.py polish in.wav out.wav --profile music
python scripts/sample_tuning.py path/to/samples
```

See `SKILL.md` for the workflow, how to read the report, and lessons learned.

MIT licence.
