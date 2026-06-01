"""Transcribe a long MP3 by splitting into ~20MB chunks and using OpenAI API.

Reads OPENAI_API_KEY from environment (do not hardcode).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openai import OpenAI

AUDIO = Path("/Users/mac/Documents/Kuliah/Semester 6/AYU PERMATA/video1894545223.mp3")
OUT_TXT = Path("/Users/mac/Documents/Kuliah/Semester 6/AYU PERMATA/video1894545223.txt")
MODEL = "gpt-4o-mini-transcribe"
LANGUAGE = "id"

# Target chunk duration. 64kbps mono/stereo MP3 ≈ 0.48MB/min, so 22 min ≈ 10.5MB.
# Keep well under the 25MB API limit.
CHUNK_SECONDS = 22 * 60


def get_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def split_audio(path: Path, out_dir: Path, seg_seconds: int) -> list[Path]:
    pattern = out_dir / "chunk_%03d.mp3"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-f", "segment",
            "-segment_time", str(seg_seconds),
            "-c", "copy",
            "-reset_timestamps", "1",
            str(pattern),
        ],
        check=True,
    )
    return sorted(out_dir.glob("chunk_*.mp3"))


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2

    client = OpenAI()

    duration = get_duration(AUDIO)
    print(f"Audio duration: {duration/60:.1f} min ({duration:.0f}s)", flush=True)

    with tempfile.TemporaryDirectory(prefix="transcribe_chunks_") as td:
        td_path = Path(td)
        print(f"Splitting into chunks of ~{CHUNK_SECONDS/60:.0f} min...", flush=True)
        chunks = split_audio(AUDIO, td_path, CHUNK_SECONDS)
        print(f"Got {len(chunks)} chunks.", flush=True)

        pieces: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            size_mb = chunk.stat().st_size / 1024 / 1024
            print(f"[{i}/{len(chunks)}] {chunk.name} ({size_mb:.1f} MB) ->", end=" ", flush=True)
            t0 = time.time()
            with chunk.open("rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=MODEL,
                    file=fh,
                    language=LANGUAGE,
                    response_format="text",
                )
            text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s, {len(text)} chars", flush=True)
            pieces.append(text.strip())

    full = "\n\n".join(pieces) + "\n"
    OUT_TXT.write_text(full, encoding="utf-8")
    print(f"\nWrote {OUT_TXT} ({len(full)} chars).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
