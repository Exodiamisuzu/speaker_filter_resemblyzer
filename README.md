# 音频分离器[Air吧]

基于 Resemblyzer 的音声识别工具，快速识别和筛选游戏/动画音频中的特定角色声纹。

## 1) 创建虚拟环境并安装

在此文件夹中打开 PowerShell，然后运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

如果 `py -3.12` 不可用，请编辑 `setup_venv.ps1` 并将 `PythonExe` 设置为有效的 Python 安装路径。

## 2) 准备参考音频

创建一个文件夹，放入**干净的目标角色音频片段**，例如：

```text
reference_clips/
  role_a_01.wav
  role_a_02.wav
  role_a_03.wav
```

然后构建参考特征向量中心点：

```powershell
python build_reference.py --reference-dir .\reference_clips --output .\models\role_a.npy --meta .\models\role_a.meta.json
```

## 3) 测试扫描并生成报告

首先运行测试扫描（不实际改动文件）：

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_scan.csv
```

`role_a_scan.csv` 字段说明：
- `score`：与目标说话人的余弦相似度
- `is_target`：1 表示预测为目标说话人
- `status`：处理状态
- `action_result`：执行的操作（或测试扫描结果）

## 4) 应用文件处理操作

### 方式 A：按前缀重命名目标文件

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_apply_rename.csv --apply --action rename_prefix --prefix ROLEA_
```

### 方式 B：将目标文件复制到另一个文件夹

```powershell
python classify_and_label.py --input-dir ..\KOE --reference-embedding .\models\role_a.npy --threshold 0.73 --report-csv .\reports\role_a_apply_copy.csv --apply --action copy_to_dir --copy-dir .\outputs\role_a_hits
```

## 阈值调整

从 `0.70 ~ 0.78` 开始：
- 误检太多 -> 增加阈值
- 漏检目标文件 -> 降低阈值

## 说明

- 支持格式：`.wav .mp3 .flac .m4a .ogg`
- 部分格式（特别是 m4a）可能需要在系统 PATH 中安装 ffmpeg
- 建议：总是先运行测试扫描，检查 CSV 报告，再执行重命名或复制操作

## GUI 应用

从虚拟环境中直接运行 GUI：

```powershell
.\run_gui.ps1
```

GUI 主要功能：
- 设置相似度阈值（0~1，默认 0.73）
- 一键构建参考特征向量（无则自动使用 Resemblyzer 内置模型）
- 扫描音频文件夹并生成识别报告
- 支持两种处理方式：复制并重命名、仅复制到输出目录
- 生成处理结果清单
- 集成运行日志和快速提示

## 打包 EXE

```powershell
.\build_exe.ps1
```

生成的可执行程序位置：

```text
dist/AirBarAudioSeparator/AirBarAudioSeparator.exe
```

打包说明：
- 默认使用 `onedir` 而非 `onefile` 以加快启动速度
- GUI 延迟加载 `torch` 和 `resemblyzer`，直到实际开始任务，窗口更快打开
- 移除未使用的 `pandas` 依赖以减小包体积
