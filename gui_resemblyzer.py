import csv
import json
import shutil
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def detect_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path(__file__).resolve().parent

    for candidate in [start, *start.parents]:
        if (candidate / "ok").exists() and (candidate / "models").exists():
            return candidate

    return start


def iter_audio_files(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def load_encoder_dependencies():
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav

    return np, VoiceEncoder, preprocess_wav


def cosine(a, b) -> float:
    np, _, _ = load_encoder_dependencies()
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
        shutil.copy2(src, candidate)
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{src.stem}__{i}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


def build_reference(reference_dir: Path, output_npy: Path, meta_json: Path, min_seconds: float, log):
    if not reference_dir.exists():
        raise FileNotFoundError(f"reference dir not found: {reference_dir}")

    log("Loading voice encoder for reference build...")
    np, VoiceEncoder, preprocess_wav = load_encoder_dependencies()
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

    np, VoiceEncoder, preprocess_wav = load_encoder_dependencies()
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
        self.root.geometry("1080x820")
        self.root.minsize(980, 760)

        self.base_dir = detect_app_base_dir()
        default_ref_dir = self.base_dir / "ok"
        default_ref_emb = self.base_dir / "models" / "ok_role.npy"
        default_ref_out = self.base_dir / "models" / "gui_role.npy"
        default_ref_meta = self.base_dir / "models" / "gui_role.meta.json"
        default_input_dir = self.base_dir / ".." / "KOE" / "0001"
        if not default_input_dir.exists():
            default_input_dir = self.base_dir / "KOE" / "0001"
        default_report = self.base_dir / "reports" / "gui_scan_report.csv"
        default_copy_dir = self.base_dir / "outputs" / "gui_target_hits"
        default_renamed = self.base_dir / "reports" / "gui_renamed_list.txt"

        self.ref_mode = IntVar(value=1)
        self.reference_dir = StringVar(value=str(default_ref_dir))
        self.reference_embedding = StringVar(value=str(default_ref_emb))
        self.reference_output_npy = StringVar(value=str(default_ref_out))
        self.reference_output_meta = StringVar(value=str(default_ref_meta))

        self.input_dir = StringVar(value=str(default_input_dir))
        self.threshold = DoubleVar(value=0.73)
        self.min_seconds = DoubleVar(value=0.8)
        self.report_csv = StringVar(value=str(default_report))

        self.apply_changes = BooleanVar(value=False)
        self.action = StringVar(value="none")
        self.prefix = StringVar(value="Misuzu")
        self.copy_dir = StringVar(value=str(default_copy_dir))
        self.renamed_txt = StringVar(value=str(default_renamed))
        self.status_text = StringVar(value="就绪")

        self.running = False
        self.start_btn = None
        self.progress_bar = None

        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#f2f5f9")
        style.configure("App.TFrame", background="#f2f5f9")
        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background="#ffffff", foreground="#12324a", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background="#f2f5f9", foreground="#0d3b66", font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background="#f2f5f9", foreground="#4a6173", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background="#f2f5f9", foreground="#5b7285", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background="#f2f5f9", foreground="#0d3b66", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=14, style="App.TFrame")
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Resemblyzer Speaker Filter", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="批量识别并标记目标角色语音", style="SubHeader.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 10))

        ref_card = ttk.LabelFrame(frame, text="1) 参考声纹来源", style="Card.TLabelframe", padding=10)
        ref_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(ref_card, text="从参考音频目录构建", variable=self.ref_mode, value=1).grid(row=0, column=0, sticky="w")
        self._add_path_row(ref_card, 1, "参考音频目录", self.reference_dir, pick_dir=True)
        self._add_path_row(ref_card, 2, "构建输出 .npy", self.reference_output_npy, pick_save=True)
        self._add_path_row(ref_card, 3, "构建输出 .meta.json", self.reference_output_meta, pick_save=True)
        ttk.Radiobutton(ref_card, text="直接使用已有 embedding (.npy)", variable=self.ref_mode, value=2).grid(row=4, column=0, sticky="w", pady=(6, 0))
        self._add_path_row(ref_card, 5, "已有 embedding", self.reference_embedding, pick_file=True)
        ref_card.grid_columnconfigure(1, weight=1)

        scan_card = ttk.LabelFrame(frame, text="2) 扫描设置", style="Card.TLabelframe", padding=10)
        scan_card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._add_path_row(scan_card, 0, "扫描目录", self.input_dir, pick_dir=True)
        self._add_path_row(scan_card, 1, "报告 CSV", self.report_csv, pick_save=True)
        ttk.Label(scan_card, text="阈值 (0~1)").grid(row=2, column=0, sticky="w", pady=(6, 2))
        ttk.Entry(scan_card, textvariable=self.threshold, width=16).grid(row=2, column=1, sticky="w", pady=(6, 2))
        ttk.Label(scan_card, text="最短秒数").grid(row=2, column=2, sticky="w", pady=(6, 2))
        ttk.Entry(scan_card, textvariable=self.min_seconds, width=16).grid(row=2, column=3, sticky="w", pady=(6, 2))
        ttk.Label(scan_card, text="建议先 dry-run，再启用文件变更", style="Hint.TLabel").grid(row=3, column=0, columnspan=4, sticky="w", pady=(2, 0))
        scan_card.grid_columnconfigure(1, weight=1)

        action_card = ttk.LabelFrame(frame, text="3) 命中文件处理", style="Card.TLabelframe", padding=10)
        action_card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Checkbutton(action_card, text="应用文件变更", variable=self.apply_changes).grid(row=0, column=0, sticky="w")
        ttk.Label(action_card, text="动作").grid(row=0, column=1, sticky="w")
        ttk.Combobox(action_card, textvariable=self.action, values=["none", "rename_prefix", "copy_to_dir"], state="readonly", width=18).grid(row=0, column=2, sticky="w")
        ttk.Label(action_card, text="重命名前缀").grid(row=1, column=0, sticky="w", pady=(6, 2))
        ttk.Entry(action_card, textvariable=self.prefix, width=24).grid(row=1, column=1, sticky="w", pady=(6, 2))
        self._add_path_row(action_card, 2, "复制输出目录", self.copy_dir, pick_dir=True)
        self._add_path_row(action_card, 3, "改名清单 TXT", self.renamed_txt, pick_save=True)
        action_card.grid_columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(frame, style="App.TFrame")
        btn_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.start_btn = ttk.Button(btn_frame, text="开始执行", style="Primary.TButton", command=self.start)
        self.start_btn.pack(side="left")
        ttk.Button(btn_frame, text="退出", command=self.root.destroy).pack(side="left", padx=8)
        self.progress_bar = ttk.Progressbar(btn_frame, mode="indeterminate", length=180)
        self.progress_bar.pack(side="left", padx=(8, 0))
        ttk.Label(btn_frame, textvariable=self.status_text, style="Status.TLabel").pack(side="left", padx=(12, 0))

        self.log_box = ScrolledText(frame, height=14, font=("Consolas", 10), bg="#fbfdff")
        self.log_box.grid(row=6, column=0, sticky="nsew")

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(6, weight=1)

    def _resolve_path(self, value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

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
        self.status_text.set("运行中...")
        if self.start_btn is not None:
            self.start_btn.configure(state="disabled")
        if self.progress_bar is not None:
            self.progress_bar.start(10)
        th = threading.Thread(target=self._run_task, daemon=True)
        th.start()

    def _run_task(self):
        try:
            self.log("========== 任务开始 ==========")
            self.log(f"Base directory: {self.base_dir}")

            if self.ref_mode.get() == 1:
                reference_embedding = build_reference(
                    self._resolve_path(self.reference_dir.get()),
                    self._resolve_path(self.reference_output_npy.get()),
                    self._resolve_path(self.reference_output_meta.get()),
                    self.min_seconds.get(),
                    self.log,
                )
            else:
                reference_embedding = self._resolve_path(self.reference_embedding.get())
                self.log(f"Using existing embedding: {reference_embedding}")

            classify_and_label(
                input_dir=self._resolve_path(self.input_dir.get()),
                reference_embedding=reference_embedding,
                threshold=self.threshold.get(),
                report_csv=self._resolve_path(self.report_csv.get()),
                min_seconds=self.min_seconds.get(),
                apply_changes=self.apply_changes.get(),
                action=self.action.get(),
                prefix=self.prefix.get(),
                copy_dir=self._resolve_path(self.copy_dir.get()),
                renamed_txt=self._resolve_path(self.renamed_txt.get()),
                log=self.log,
            )

            self.log("========== 任务完成 ==========")
            self.status_text.set("完成")
            messagebox.showinfo("完成", "处理完成，请查看日志与输出文件")
        except Exception as ex:
            self.log(f"ERROR: {ex}")
            self.status_text.set("失败")
            messagebox.showerror("执行失败", str(ex))
        finally:
            self.running = False
            if self.start_btn is not None:
                self.start_btn.configure(state="normal")
            if self.progress_bar is not None:
                self.progress_bar.stop()


def main():
    root = Tk()
    app = App(root)
    app.log("GUI ready. Configure paths and click 开始执行")
    root.mainloop()


if __name__ == "__main__":
    main()
