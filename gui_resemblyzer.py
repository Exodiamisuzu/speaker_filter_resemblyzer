import csv
import json
import threading
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

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

    i = 1
    while True:
        candidate = path.parent / f"{prefix}{path.stem}__{i}{path.suffix}"
        if not candidate.exists():
            return path.rename(candidate)
        i += 1


def safe_copy_to_folder(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / src.name
    if not candidate.exists():
        candidate.write_bytes(src.read_bytes())
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{src.stem}__{i}{src.suffix}"
        if not candidate.exists():
            candidate.write_bytes(src.read_bytes())
            return candidate
        i += 1


def build_reference(reference_dir: Path, output_npy: Path, meta_json: Path, min_seconds: float, log):
    if not reference_dir.exists():
        raise FileNotFoundError(f"reference dir not found: {reference_dir}")

    log("Loading voice encoder for reference build...")
    encoder = VoiceEncoder()
    embeddings = []
    used_files = []
    skipped_files = []

    files = list(iter_audio_files(reference_dir))
    log(f"Reference audio files found: {len(files)}")

    for idx, audio_path in enumerate(files, start=1):
        try:
            wav = preprocess_wav(str(audio_path))
            seconds = len(wav) / 16000.0
            if seconds < min_seconds:
                skipped_files.append({"path": str(audio_path), "reason": f"too_short<{min_seconds}s"})
                continue
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            used_files.append(str(audio_path))
        except Exception as ex:
            skipped_files.append({"path": str(audio_path), "reason": str(ex)})

        if idx % 20 == 0:
            log(f"Reference progress: {idx}/{len(files)}")

    if not embeddings:
        raise RuntimeError("No valid reference audio found. Please provide more clean clips.")

    centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, centroid)

    meta = {
        "reference_dir": str(reference_dir),
        "used_count": len(used_files),
        "skipped_count": len(skipped_files),
        "used_files": used_files,
        "skipped_files": skipped_files,
        "embedding_shape": list(centroid.shape),
        "sample_rate": 16000,
    }
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log(f"Saved centroid embedding: {output_npy}")
    log(f"Saved metadata: {meta_json}")
    log(f"Reference used {len(used_files)} files, skipped {len(skipped_files)} files")

    return output_npy


def classify_and_label(
    input_dir: Path,
    reference_embedding: Path,
    threshold: float,
    report_csv: Path,
    min_seconds: float,
    apply_changes: bool,
    action: str,
    prefix: str,
    copy_dir: Path,
    renamed_txt: Path,
    log,
):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not reference_embedding.exists():
        raise FileNotFoundError(f"reference embedding not found: {reference_embedding}")

    ref = np.load(reference_embedding)
    ref = ref / (np.linalg.norm(ref) + 1e-9)

    log("Loading voice encoder for scan...")
    encoder = VoiceEncoder()

    all_files = list(iter_audio_files(input_dir))
    log(f"Scanning file count: {len(all_files)}")

    rows = []
    target_count = 0
    skipped_count = 0
    renamed_paths = []

    for idx, audio_path in enumerate(all_files, start=1):
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

            if seconds < min_seconds:
                row["status"] = f"skipped_short<{min_seconds}s"
                skipped_count += 1
                rows.append(row)
                continue

            emb = encoder.embed_utterance(wav)
            score = cosine(emb, ref)
            row["score"] = f"{score:.6f}"

            is_target = score >= threshold
            row["is_target"] = 1 if is_target else 0

            if is_target:
                target_count += 1
                if apply_changes and action == "rename_prefix":
                    new_path = safe_rename_with_prefix(audio_path, prefix)
                    row["action_result"] = f"renamed_to:{new_path}"
                    renamed_paths.append(str(new_path))
                elif apply_changes and action == "copy_to_dir":
                    out = safe_copy_to_folder(audio_path, copy_dir)
                    row["action_result"] = f"copied_to:{out}"
                else:
                    row["action_result"] = "dry_run_or_no_action"
        except Exception as ex:
            row["status"] = f"error:{str(ex)}"

        rows.append(row)

        if idx % 100 == 0:
            log(f"Scan progress: {idx}/{len(all_files)}")

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "seconds", "score", "is_target", "status", "action_result"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if renamed_txt and renamed_paths:
        renamed_txt.parent.mkdir(parents=True, exist_ok=True)
        renamed_txt.write_text("\n".join(renamed_paths), encoding="utf-8")

    log(f"Scanned files: {len(all_files)}")
    log(f"Target files (score >= {threshold}): {target_count}")
    log(f"Skipped short files: {skipped_count}")
    log(f"Report written to: {report_csv}")
    if renamed_txt and renamed_paths:
        log(f"Renamed list written to: {renamed_txt} ({len(renamed_paths)} items)")


class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Resemblyzer Speaker Filter")
        self.root.geometry("980x760")

        self.ref_mode = IntVar(value=1)
        self.reference_dir = StringVar(value="ok")
        self.reference_embedding = StringVar(value="models/ok_role.npy")
        self.reference_output_npy = StringVar(value="models/gui_role.npy")
        self.reference_output_meta = StringVar(value="models/gui_role.meta.json")

        self.input_dir = StringVar(value="../KOE/0001")
        self.threshold = DoubleVar(value=0.73)
        self.min_seconds = DoubleVar(value=0.8)
        self.report_csv = StringVar(value="reports/gui_scan_report.csv")

        self.apply_changes = BooleanVar(value=False)
        self.action = StringVar(value="none")
        self.prefix = StringVar(value="Misuzu")
        self.copy_dir = StringVar(value="outputs/gui_target_hits")
        self.renamed_txt = StringVar(value="reports/gui_renamed_list.txt")

        self.running = False

        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="1) 参考声纹来源", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

        ttk.Radiobutton(frame, text="从参考音频目录构建", variable=self.ref_mode, value=1).grid(row=1, column=0, sticky="w")
        self._add_path_row(frame, 2, "参考音频目录", self.reference_dir, pick_dir=True)
        self._add_path_row(frame, 3, "构建输出 .npy", self.reference_output_npy, pick_save=True)
        self._add_path_row(frame, 4, "构建输出 .meta.json", self.reference_output_meta, pick_save=True)

        ttk.Radiobutton(frame, text="直接使用已有 embedding (.npy)", variable=self.ref_mode, value=2).grid(row=5, column=0, sticky="w", pady=(8, 0))
        self._add_path_row(frame, 6, "已有 embedding", self.reference_embedding, pick_file=True)

        ttk.Separator(frame).grid(row=7, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Label(frame, text="2) 扫描设置", font=("Segoe UI", 11, "bold")).grid(row=8, column=0, sticky="w", pady=(0, 6))
        self._add_path_row(frame, 9, "扫描目录", self.input_dir, pick_dir=True)
        self._add_path_row(frame, 10, "报告 CSV", self.report_csv, pick_save=True)

        ttk.Label(frame, text="阈值 (0~1)").grid(row=11, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.threshold, width=14).grid(row=11, column=1, sticky="w")

        ttk.Label(frame, text="最短秒数").grid(row=11, column=2, sticky="w")
        ttk.Entry(frame, textvariable=self.min_seconds, width=14).grid(row=11, column=3, sticky="w")

        ttk.Separator(frame).grid(row=12, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Label(frame, text="3) 命中文件处理", font=("Segoe UI", 11, "bold")).grid(row=13, column=0, sticky="w", pady=(0, 6))

        ttk.Checkbutton(frame, text="应用文件变更", variable=self.apply_changes).grid(row=14, column=0, sticky="w")

        ttk.Label(frame, text="动作").grid(row=14, column=1, sticky="w")
        ttk.Combobox(frame, textvariable=self.action, values=["none", "rename_prefix", "copy_to_dir"], state="readonly", width=18).grid(row=14, column=2, sticky="w")

        ttk.Label(frame, text="重命名前缀").grid(row=15, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.prefix, width=25).grid(row=15, column=1, sticky="w")

        self._add_path_row(frame, 16, "复制输出目录", self.copy_dir, pick_dir=True)
        self._add_path_row(frame, 17, "改名清单 TXT", self.renamed_txt, pick_save=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=18, column=0, columnspan=4, sticky="w", pady=(10, 8))
        ttk.Button(btn_frame, text="开始执行", command=self.start).pack(side="left")
        ttk.Button(btn_frame, text="退出", command=self.root.destroy).pack(side="left", padx=8)

        self.log_box = ScrolledText(frame, height=16)
        self.log_box.grid(row=19, column=0, columnspan=4, sticky="nsew")

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(19, weight=1)

    def _add_path_row(self, parent, row, label, var, pick_dir=False, pick_file=False, pick_save=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=80).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=2)

        def choose():
            if pick_dir:
                value = filedialog.askdirectory()
            elif pick_file:
                value = filedialog.askopenfilename(filetypes=[("NumPy file", "*.npy"), ("All files", "*.*")])
            elif pick_save:
                value = filedialog.asksaveasfilename()
            else:
                value = ""
            if value:
                var.set(value)

        ttk.Button(parent, text="浏览", command=choose).grid(row=row, column=3, sticky="w")

    def log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {text}\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def _validate(self):
        if self.threshold.get() < 0 or self.threshold.get() > 1:
            raise ValueError("阈值必须在 0 到 1 之间")
        if self.min_seconds.get() <= 0:
            raise ValueError("最短秒数必须大于 0")

        if self.apply_changes.get():
            if self.action.get() not in {"rename_prefix", "copy_to_dir"}:
                raise ValueError("已勾选应用文件变更时，动作必须是 rename_prefix 或 copy_to_dir")
            if self.action.get() == "rename_prefix" and not self.prefix.get().strip():
                raise ValueError("重命名前缀不能为空")

    def start(self):
        if self.running:
            messagebox.showinfo("提示", "任务正在执行中，请稍候")
            return

        try:
            self._validate()
        except Exception as ex:
            messagebox.showerror("参数错误", str(ex))
            return

        self.running = True
        th = threading.Thread(target=self._run_task, daemon=True)
        th.start()

    def _run_task(self):
        try:
            self.log("========== 任务开始 ==========")

            if self.ref_mode.get() == 1:
                reference_embedding = build_reference(
                    Path(self.reference_dir.get()),
                    Path(self.reference_output_npy.get()),
                    Path(self.reference_output_meta.get()),
                    self.min_seconds.get(),
                    self.log,
                )
            else:
                reference_embedding = Path(self.reference_embedding.get())
                self.log(f"Using existing embedding: {reference_embedding}")

            classify_and_label(
                input_dir=Path(self.input_dir.get()),
                reference_embedding=reference_embedding,
                threshold=self.threshold.get(),
                report_csv=Path(self.report_csv.get()),
                min_seconds=self.min_seconds.get(),
                apply_changes=self.apply_changes.get(),
                action=self.action.get(),
                prefix=self.prefix.get(),
                copy_dir=Path(self.copy_dir.get()),
                renamed_txt=Path(self.renamed_txt.get()),
                log=self.log,
            )

            self.log("========== 任务完成 ==========")
            messagebox.showinfo("完成", "处理完成，请查看日志与输出文件")
        except Exception as ex:
            self.log(f"ERROR: {ex}")
            messagebox.showerror("执行失败", str(ex))
        finally:
            self.running = False


def main():
    root = Tk()
    app = App(root)
    app.log("GUI ready. Configure paths and click 开始执行")
    root.mainloop()


if __name__ == "__main__":
    main()
