# AI辅助说明：Kimi K2.6，v2.6，Moonshot AI，2026-07-29，
# 生成/辅助内容：台式机一键训练脚本，自动安装依赖、选择 GPU/CPU 模式、运行任务一/二训练、
# 检查结果并自动提交结果到本地仓库。适用于 Windows PowerShell 环境。
# 该内容经作者核对修改后用于 B 题比赛桌面部署。

# ============================================================
# 使用说明
# ============================================================
# 1. 在台式机上：
#      git clone https://github.com/blackwhitetony/innoCupTaskB.git
#      cd innoCupTaskB
#
# 2. CPU 模式（默认）：
#      .\run_training.ps1
#
# 3. GPU 模式：
#      .\run_training.ps1 -Mode GPU
#
# 4. 只跑任务二：
#      .\run_training.ps1 -Task2
#
# 5. 不自动提交结果：
#      .\run_training.ps1 -NoCommit
#
# 6. 训练结束后手动推送：
#      git push
# ============================================================

param(
    [ValidateSet("CPU", "GPU")]
    [string]$Mode = "CPU",
    [switch]$Task1,
    [switch]$Task2,
    [switch]$NoCommit
)

$ErrorActionPreference = "Stop"

function Test-GPU {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warning "nvidia-smi 未找到，GPU 不可用。将自动回退到 CPU 模式。"
        return $false
    }
    try {
        $smiOutput = nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($smiOutput) {
            Write-Host "检测到 GPU: $smiOutput" -ForegroundColor Green
            return $true
        }
    } catch { }
    Write-Warning "nvidia-smi 检测失败，将回退到 CPU 模式。"
    return $false
}

# ============================================================
# 1. 检查 uv
# ============================================================
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv 未安装，正在下载安装..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv 安装失败，请手动安装后重试。"
    }
    Write-Host "uv 安装成功" -ForegroundColor Green
} else {
    Write-Host "uv 已安装: $(uv --version)" -ForegroundColor Green
}

# ============================================================
# 2. 安装依赖
# ============================================================
Write-Host "`n正在安装 Python 依赖（锁定版本）..." -ForegroundColor Yellow
uv sync
Write-Host "依赖安装完成" -ForegroundColor Green

# ============================================================
# 3. GPU 模式校验
# ============================================================
if ($Mode -eq "GPU") {
    $gpuAvailable = Test-GPU
    if (-not $gpuAvailable) {
        Write-Warning "GPU 不可用，自动回退到 CPU 模式"
        $Mode = "CPU"
    }
}

$env:CATBOOST_TASK = $Mode
Write-Host "`nCatBoost 运行模式: $Mode" -ForegroundColor Cyan

# ============================================================
# 4. 运行训练
# ============================================================
$tasks = @()
if (-not $Task1 -and -not $Task2) {
    $tasks = @("task1", "task2")
} else {
    if ($Task1) { $tasks += "task1" }
    if ($Task2) { $tasks += "task2" }
}

foreach ($task in $tasks) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  开始训练: $task" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    uv run python "src/train_$task.py"
}

# ============================================================
# 5. 检查结果
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  检查结果" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$modelDir = "$PSScriptRoot\models"
$outputDir = "$PSScriptRoot\outputs"

$modelFiles = Get-ChildItem -Path $modelDir -Filter "*.cbm" -ErrorAction SilentlyContinue
$outputFiles = Get-ChildItem -Path $outputDir -Filter "*.csv" -ErrorAction SilentlyContinue

if ($modelFiles) {
    Write-Host "模型文件:" -ForegroundColor Green
    foreach ($f in $modelFiles) { Write-Host "  - $($f.Name)" }
} else {
    Write-Warning "未找到 .cbm 模型文件"
}

if ($outputFiles) {
    Write-Host "`n预测结果:" -ForegroundColor Green
    foreach ($f in $outputFiles) { Write-Host "  - $($f.Name)" }
} else {
    Write-Warning "未找到 CSV 结果文件"
}

# ============================================================
# 6. 自动提交结果（可选）
# ============================================================
if (-not $NoCommit) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Git 提交结果" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # 只提交训练产物，不提交 catboost_info 日志（已 gitignore）
    git add models/ outputs/ src/ features.py docs/ run_training.ps1 AGENTS.md
    git commit -m "train: $Mode 模式训练结果 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "结果已提交到本地仓库。运行 git push 推送到远程。" -ForegroundColor Green
} else {
    Write-Host "`n跳过 Git 提交（-NoCommit 模式）" -ForegroundColor Yellow
}

Write-Host "`n全部完成！" -ForegroundColor Green
