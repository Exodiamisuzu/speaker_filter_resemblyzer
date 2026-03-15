import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def iter_audio_files(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def safe_rename_with_prefix(path: Path, prefix: str) -> Path:
    candidate = path.with_name(f"{prefix}{path.name}")
    if not candidate.exists():
        return path.rename(candidate)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{prefix}{stem}__{i}{suffix}"
        if not candidate.exists():
            return path.rename(candidate)
        i += 1


def safe_copy_to_folder(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / src.name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{src.stem}__{i}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="Classify and label target-speaker audio by cosine similarity with reference embedding."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder to scan")
    parser.add_argument("--reference-embedding", required=True, type=Path, help=".npy file created by build_reference.py")
    parser.add_argument("--threshold", type=float, default=0.73, help="Similarity threshold to mark as target")
    parser.add_argument("--report-csv", default=Path("scan_report.csv"), type=Path, help="Output CSV report")
    parser.add_argument("--min-seconds", type=float, default=0.8, help="Skip clips shorter than this after preprocessing")

    parser.add_argument("--apply", action="store_true", help="Actually apply file labeling action")
    parser.add_argument("--action", choices=["rename_prefix", "copy_to_dir", "none"], default="none", help="What to do for target files")
    parser.add_argument("--prefix", default="TARGET_", help="Prefix used when action=rename_prefix")
    parser.add_argument("--copy-dir", type=Path, default=Path("target_hits"), help="Destination when action=copy_to_dir")

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {args.input_dir}")
    if not args.reference_embedding.exists():
        raise FileNotFoundError(f"reference embedding not found: {args.reference_embedding}")

    ref = np.load(args.reference_embedding)
    ref = ref / (np.linalg.norm(ref) + 1e-9)

    encoder = VoiceEncoder()

    all_files = list(iter_audio_files(args.input_dir))
    rows = []
    target_count = 0
    skipped_count = 0

    for audio_path in tqdm(all_files, desc="Scanning"):
        row = {
            "path": str(audio_path),
            "seconds": "",
            "score": "",
            "is_target": 0,
            "status": "ok",
            "action_result": "",
        }
        try:
            wav = preprocess_wav(str(audio_path))
            seconds = len(wav) / 16000.0
            row["seconds"] = f"{seconds:.3f}"

            if seconds < args.min_seconds:
                row["status"] = f"skipped_short<{args.min_seconds}s"
                skipped_count += 1
                rows.append(row)
                continue

            emb = encoder.embed_utterance(wav)
            score = cosine(emb, ref)
            row["score"] = f"{score:.6f}"

            is_target = score >= args.threshold
            row["is_target"] = 1 if is_target else 0

            if is_target:
                target_count += 1
                if args.apply and args.action == "rename_prefix":
                    new_path = safe_rename_with_prefix(audio_path, args.prefix)
                    row["action_result"] = f"renamed_to:{new_path}"
                elif args.apply and args.action == "copy_to_dir":
                    out = safe_copy_to_folder(audio_path, args.copy_dir)
                    row["action_result"] = f"copied_to:{out}"
                else:
                    row["action_result"] = "dry_run_or_no_action"

        except Exception as ex:
            row["status"] = f"error:{str(ex)}"

        rows.append(row)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "seconds", "score", "is_target", "status", "action_result"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned files: {len(all_files)}")
    print(f"Target files (score >= {args.threshold}): {target_count}")
    print(f"Skipped short files: {skipped_count}")
    print(f"Report written to: {args.report_csv}")


if __name__ == "__main__":
    main()
