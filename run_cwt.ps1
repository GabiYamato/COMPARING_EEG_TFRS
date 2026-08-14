# ============================================================
# run_cwt.ps1
# Runs missing models (cCNN ablations, cRNN ablations, new models, reference models)
# on the CWT spectrogram dataset. Automatically SKIPS models
# that have already completed!
# ============================================================

$ErrorActionPreference = "Stop"
$SCRIPT  = Join-Path $PSScriptRoot "main.py"
$DATASET = "cwt"
$BASE_DIR = Join-Path $PSScriptRoot $DATASET

function Run-Model {
    param([string]$ModelName, [string]$Mode)
    
    $ci_file = Join-Path $BASE_DIR "metrics\$ModelName\summary_ci.json"
    if (Test-Path $ci_file) {
        Write-Host "  [SKIPPED] $ModelName already completed." -ForegroundColor DarkGray
        return
    }

    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host "  DATASET : $DATASET   MODEL : $ModelName" -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Cyan
    python $SCRIPT --dataset $DATASET --mode $Mode --model $ModelName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAILED] $ModelName exited with code $LASTEXITCODE" -ForegroundColor Red
    } else {
        Write-Host "  [DONE]   $ModelName finished successfully." -ForegroundColor Green
    }
}

$start = Get-Date
Write-Host "====================================================" -ForegroundColor Yellow
Write-Host "  Starting CWT run  ($start)" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Yellow

# ---- cCNN Ablations (A1-A6) ----
Write-Host "`n### cCNN ABLATIONS ###" -ForegroundColor Magenta
Run-Model "cCNN"                 "ablation"
Run-Model "cCNN_A1_NoPooling"    "ablation"
Run-Model "cCNN_A2_NoDropout"    "ablation"
Run-Model "cCNN_A3_Dense64"      "ablation"
Run-Model "cCNN_A4_Filters16"    "ablation"
Run-Model "cCNN_A5_Filters64"    "ablation"
Run-Model "cCNN_A6_Kernel5"      "ablation"

# ---- cRNN Ablations (B1-B6) ----
Write-Host "`n### cRNN ABLATIONS ###" -ForegroundColor Magenta
Run-Model "cRNN"                 "ablation"
Run-Model "cRNN_B1_OneLayer"     "ablation"
Run-Model "cRNN_B2_TwoLayers"    "ablation"
Run-Model "cRNN_B3_NoDropout"    "ablation"
Run-Model "cRNN_B4_Hidden1024"   "ablation"
Run-Model "cRNN_B5_Hidden512"    "ablation"
Run-Model "cRNN_B6_GRU"          "ablation"

# ---- New Models ----
Write-Host "`n### NEW MODELS ###" -ForegroundColor Magenta
Run-Model "CNN_LSTM"             "new"
Run-Model "CNN_BiLSTM"           "new"
Run-Model "Swin_Transformer"     "new"

# ---- Reference / Additional Models ----
Write-Host "`n### REFERENCE / ADDITIONAL MODELS ###" -ForegroundColor Magenta
Run-Model "AlexNet"              "models"
Run-Model "VGGNET"               "models"
Run-Model "LSTM"                 "models"
Run-Model "ResBiLSTM"            "models"

# ---- Regenerate Summary CSVs and Notebooks ----
Write-Host "`nUpdating summary CSVs and notebooks..." -ForegroundColor Cyan
python (Join-Path $PSScriptRoot "csv_generator.py")
python (Join-Path $PSScriptRoot "notebook_generator.py")

# ---- Done ----
$end = Get-Date
$elapsed = $end - $start
Write-Host ""
Write-Host "====================================================" -ForegroundColor Yellow
Write-Host "  CWT run complete." -ForegroundColor Yellow
Write-Host "  Total time: $([math]::Round($elapsed.TotalMinutes, 1)) minutes" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Yellow
