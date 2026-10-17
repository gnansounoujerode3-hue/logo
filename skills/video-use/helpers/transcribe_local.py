"""Transcribe a video locally with faster-whisper (free, no API key).

Fallback for helpers/transcribe.py (ElevenLabs Scribe) once the free credits are used up.
Writes the same JSON shape Scribe returns (words + spacing entries, word timestamps),
so pack_transcripts.py and the rest of the pipeline read it unchanged.

Usage:
    pip install faster-whisper
    python helpers/transcribe_local.py <video_path>
    python helpers/transcribe_local.py <video_path> --model small --language fr
    python helpers/transcribe_local.py <video_path> --edit-dir /custom/edit

Models (downloaded once from HuggingFace, then fully offline): tiny, base, small, medium,
large-v3. `small` is a reasonable default on CPU; `base` is faster on weak machines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from transcribe import transcript_path  # same cache naming as the Scribe helper


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
        "-map", f"0:a:{audio_track}", "-ac", "1", "-ar", "16000", str(dest),
    ]
    subprocess.run(cmd, check=True)


def to_scribe_shape(segments, language: str | None, duration: float) -> dict:
    """Convert faster-whisper segments to the Scribe-like layout used downstream."""
    words: list[dict] = []
    prev_end: float | None = None
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            start, end = float(w.start), float(w.end)
            if prev_end is not None and start > prev_end + 0.001:
                words.append({"text": " ", "type": "spacing", "start": prev_end, "end": start})
            words.append({"text": text, "type": "word", "start": start, "end": end, "speaker_id": None})
            prev_end = end
    return {
        "language_code": language,
        "language_probability": None,
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "audio_duration_secs": duration,
        "source": "faster-whisper",
    }


def transcribe_local(video: Path, edit_dir: Path, model_size: str, language: str | None,
                     audio_track: int, verbose: bool = True) -> Path:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed: pip install faster-whisper")

    out_path = transcript_path(edit_dir, video, audio_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        if verbose:
            print(f"  extracting audio from {video.name}", flush=True)
        extract_audio(video, audio, audio_track)

        if verbose:
            print(f"  loading model '{model_size}' (CPU, int8)", flush=True)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(
            str(audio), language=language, word_timestamps=True, vad_filter=True,
        )
        segments = list(segments_iter)  # consume the generator

    payload = to_scribe_shape(segments, info.language, info.duration)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if verbose:
        n_words = sum(1 for w in payload["words"] if w["type"] == "word")
        print(f"  saved: {out_path.name} — {n_words} words, language {info.language}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Local transcription with faster-whisper (free fallback)")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="defaults to <video folder>/edit")
    ap.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    ap.add_argument("--language", default=None, help="e.g. fr, en (auto-detect if omitted)")
    ap.add_argument("--audio-track", type=int, default=0)
    args = ap.parse_args()

    edit_dir = args.edit_dir or (args.video.parent / "edit")
    transcribe_local(args.video, edit_dir, args.model, args.language, args.audio_track)


if __name__ == "__main__":
    main()
