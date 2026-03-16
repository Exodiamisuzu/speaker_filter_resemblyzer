# Speaker Filter with Resemblyzer

This mini project helps you identify one target character/speaker from many game audio files.

## 1) Create venv and install

Open PowerShell in this folder, then run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

If `py -3.12` is not available, edit `setup_venv.ps1` and set `PythonExe` to a valid Python.

## 2) Prepare reference clips

Create a folder with **clean clips** of the target character, for example:

```text
reference_clips/
  role_a_01.wav
  role_a_02.wav
  role_a_03.wav
```

Then build reference embedding centroid:

```powershell
python build_reference.py --reference-dir .\reference_clips --output .\models\role_a.npy --meta .\models\role_a.meta.json
```

## 3) Dry-run scan and generate report

Run dry-run first (no file changes):

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_scan.csv
```

`role_a_scan.csv` fields:
- `score`: cosine similarity with target speaker
- `is_target`: 1 means predicted as target
- `status`: processing status
- `action_result`: what action was performed (or dry-run)

## 4) Apply file labeling action

### Option A: Rename target files by prefix

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_apply_rename.csv --apply --action rename_prefix --prefix ROLEA_
```

### Option B: Copy target files to another folder

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_apply_copy.csv --apply --action copy_to_dir --copy-dir .\outputs\role_a_hits
```

## Threshold tuning

Start with `0.70 ~ 0.78`.
- Too many false positives -> increase threshold
- Missing true target files -> decrease threshold

## Notes

- Supports: `.wav .mp3 .flac .m4a .ogg`
- For some formats (especially m4a), ffmpeg may be required in PATH.
- Recommended: always run dry-run first, check CSV, then apply rename/copy.

## GUI app

Run GUI directly from venv:

```powershell
.\run_gui.ps1
```

GUI supports:
- Set threshold (`0~1`)
- Set input folder and report csv output path
- Choose whether to apply file changes
- Control rename prefix
- Control copy output folder
- Output renamed file list to txt

## Build EXE

```powershell
.\build_exe.ps1
```

Then executable is generated at:

```text
dist/ResemblyzerSpeakerFilter/ResemblyzerSpeakerFilter.exe
```

Packaging notes:
- Default build now uses `onedir` instead of `onefile` for faster startup.
- GUI delays loading `torch` and `resemblyzer` until you actually start a task, so the window opens faster.
- Unused `pandas` dependency is removed to avoid unnecessary package size.

## Build Compact EXE

If you prefer a single-file portable build with smaller total distribution size, run:

```powershell
.\build_exe_compact.ps1
```

Then executable is generated at:

```text
dist/ResemblyzerSpeakerFilter_compact.exe
```

Trade-off:
- `build_exe.ps1`: faster startup, but folder distribution is larger.
- `build_exe_compact.ps1`: smaller to carry around, but startup is slower because onefile needs extraction.
