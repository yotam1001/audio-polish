"""Voice processing for recorded voice-over takes and a simple music mix.

  python voice_chain.py take <in.wav> <out.wav>          one take -> clean, level, -16 LUFS
  python voice_chain.py mix <video.mp4> <music.wav> <out.mp4> [music_db]
      voice-only render + music bed: music ducks under the voice, master -14 LUFS / -1.5 dBTP

Why each step (from tools/voice_report.py on the Nov 23 takes):
- headset proximity makes <120 Hz and 200-500 Hz heavy while 2-5 kHz is ~15 dB down -> HPF, mud cut, presence lift
- words fade inside a line (~20 dB p10/p90 spread) -> speechnorm lifts quiet syllables, then a gentle compressor
- raw takes span -21..-30 LUFS -> two-pass linear loudnorm to the same target per line
"""
import json
import subprocess
import sys

CHAIN = ",".join([
    "adeclip",
    "highpass=f=90:poles=2",
    "afftdn=nr=10:nf=-58:tn=1",
    "agate=threshold=0.004:ratio=2:range=0.2:attack=8:release=220:knee=4",
    "equalizer=f=250:t=q:w=1.1:g=-4",
    "equalizer=f=3200:t=q:w=1.4:g=3.5",
    "highshelf=f=10000:g=0.5",
    "deesser=i=0.6:m=0.6:f=0.5",
    "speechnorm=e=3:r=0.0002:l=1",
    "acompressor=threshold=-22dB:ratio=3:attack=6:release=90:makeup=2",
    "alimiter=limit=0.7:attack=3:release=40:level=false",
])


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True)


def measure(src, pre):
    o = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", pre + ",loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    return json.loads(o[o.rfind("{"):o.rfind("}") + 1])


def take(src, dst, target=-16.0):
    m = measure(src, CHAIN)
    ln = (f"loudnorm=I={target}:TP=-1.5:LRA=7:linear=true:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", CHAIN + "," + ln, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst])
    got = float(measure(dst, "anull")["input_i"])
    if got < target - 0.7:  # peak-limited line: lift with a limiter instead of leaving it quiet
        tmp = dst + ".lift.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", dst, "-af", f"volume={target - got:.2f}dB,alimiter=limit=0.84:attack=2:release=40:level=false",
             "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", tmp])
        import os
        os.replace(tmp, dst)


def mix(video, music, out, music_db=-17.0):
    # voice = the render's own audio; music ducks ~9 dB under it with a smooth release
    fc = (
        f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[v][key];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={music_db}dB[m];"
        f"[m][key]sidechaincompress=threshold=0.02:ratio=8:attack=40:release=450:makeup=1[md];"
        f"[v][md]amix=inputs=2:duration=first:normalize=0[pre]"
    )
    tmp = out + ".premix.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", music, "-filter_complex", fc, "-map", "[pre]", "-ar", "48000", tmp])
    m = measure(tmp, "anull")
    ln = (f"loudnorm=I=-14:TP=-1.5:LRA=9:linear=true:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}")
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", tmp, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-af", ln + ",alimiter=limit=0.84:level=false", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", out])
    import os
    os.remove(tmp)


if __name__ == "__main__":
    if sys.argv[1] == "take":
        take(sys.argv[2], sys.argv[3])
    else:
        mix(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else -17.0)
