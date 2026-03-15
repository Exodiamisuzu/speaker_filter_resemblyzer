import argparse
import json
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def iter_audio_files(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(
        description="Build a reference embedding centroid for one target character/speaker."
    )
    parser.add_argument("--reference-dir", required=True, type=Path, help="Folder containing clean reference clips of target speaker")
    parser.add_argument("--output", default=Path("reference_embedding.npy"), type=Path, help="Output .npy centroid file")
    parser.add_argument("--meta", default=Path("reference_embedding.meta.json"), type=Path, help="Output metadata json")
    parser.add_argument("--min-seconds", type=float, default=0.8, help="Skip clips shorter than this after preprocessing")
    args = parser.parse_args()

    ref_dir: Path = args.reference_dir
    if not ref_dir.exists():
        raise FileNotFoundError(f"reference dir not found: {ref_dir}")

    encoder = VoiceEncoder()
    embeddings = []
    used_files = []
    skipped_files = []

    for audio_path in iter_audio_files(ref_dir):
        try:
            wav = preprocess_wav(str(audio_path))
            seconds = len(wav) / 16000.0
            if seconds < args.min_seconds:
                skipped_files.append({"path": str(audio_path), "reason": f"too_short<{args.min_seconds}s"})
                continue
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            used_files.append(str(audio_path))
        except Exception as ex:
            skipped_files.append({"path": str(audio_path), "reason": str(ex)})

    if not embeddings:
        raise RuntimeError("No valid reference audio found. Please provide more clean clips.")

    centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, centroid)

    meta = {
        "reference_dir": str(ref_dir),
        "used_count": len(used_files),
        "skipped_count": len(skipped_files),
        "used_files": used_files,
        "skipped_files": skipped_files,
        "embedding_shape": list(centroid.shape),
        "sample_rate": 16000,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved centroid embedding: {args.output}")
    print(f"Saved metadata: {args.meta}")
    print(f"Used {len(used_files)} files, skipped {len(skipped_files)} files")


if __name__ == "__main__":
    main()
