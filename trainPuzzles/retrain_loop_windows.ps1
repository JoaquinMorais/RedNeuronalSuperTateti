# ==========================================================================
# retrain_loop_windows.ps1 -- Reintento automatico de "mate en N" con MAS
# DATOS, nativo de Windows (PowerShell + Docker Desktop con GPU/WSL2).
#
#   powershell -ExecutionPolicy Bypass -File trainPuzzles/retrain_loop_windows.ps1
#
# Entrena SIEMPRE con GPU (torch cu128, docker-compose.gpu.yml). Hace lo mismo
# que trainPuzzles/retrain_loop.sh (Linux/CPU):
#   * Se ubica solo en la raiz del repo (no importa desde donde lo lances).
#   * Construye una vez la imagen GPU y valida que Docker responde.
#   * Ronda 1 arranca desde CANON_INIT con PPL puzzles por nivel. Si el nivel
#     objetivo no queda 'mastered', reintenta con mas puzzles (+PPL_STEP),
#     semilla de train nueva y --eval-seed FIJO, reanudando desde el canon.
#   * Cada ronda archiva en <OUTDIR>/runs/NNNN_retryN/ y actualiza el canon.
#
# Requiere Docker Compose v2.30+ (soporte de `gpus`) y Docker Desktop con
# integracion WSL2 + GPU. Verifica con:
#   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# ==========================================================================
[CmdletBinding()]
param(
    [int]$ROUNDS = 5,
    [int]$TARGET_LEVEL = 2,
    [int]$PPL = 25000,
    [int]$PPL_STEP = 25000,
    [int]$MAX_EPOCHS = 3000,
    [int]$PLATEAU = 1500,
    [int]$SEED = 0,
    [int]$EVAL_SEED = 0,
    [double]$DROPOUT = 0.2,
    [string]$WEIGHT_DECAY = "1e-4",
    [int]$WARM_START = 1,
    [double]$THRESHOLD = 0.90,
    [int]$MAX_LEVEL = 3,
    [int]$START_LEVEL = 1,
    [ValidateSet("mlp", "cnn")][string]$ARCH = "mlp",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "cuda",

    # Tecnicas de entrenamiento (mismos nombres que el .sh)
    [ValidateSet("none", "cosine")][string]$LR_SCHEDULE = "none",
    [double]$LR_FINAL_FRAC = 0.03,
    [int]$TTA = 0,
    [int]$EVAL_EVERY = 1,
    [int]$HARD_MINING = 0,
    [int]$MINE_EVERY = 10,
    [double]$MINE_BOOST = 3.0,
    [int]$MINE_WARMUP = 25,
    [int]$EMA = 0,
    [double]$EMA_DECAY = 0.999,
    [double]$RANK_COEF = 0.0,
    [double]$RANK_MARGIN = 1.0,

    [string]$CANON_INIT = "",
    [string]$OUTDIR = ""
)

$ErrorActionPreference = "Stop"

# --- 0.0 Defaults dependientes del arch -----------------------------------
# Vacios => derivar de $ARCH (mlp: checkpoints_mts, cnn: checkpoint_cnn).
if ([string]::IsNullOrEmpty($OUTDIR)) {
    if ($ARCH -eq "cnn") { $OUTDIR = "trainPuzzles/checkpoint_cnn" }
    else { $OUTDIR = "trainPuzzles/checkpoints_mts" }
}
if ([string]::IsNullOrEmpty($CANON_INIT)) { $CANON_INIT = "$OUTDIR/puzzle_canon.pth" }

# --- 0. Helpers -----------------------------------------------------------
# Los floats se interpolan con la cultura actual: en Windows en espanol
# [double]0.2 -> "0,2" y rompe el CLI de Python. Formateamos siempre invariante.
$script:inv = [System.Globalization.CultureInfo]::InvariantCulture
function InvFmt([double]$v) { return $v.ToString($script:inv) }

# --- 1. Ubicarse en la raiz del repo --------------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "== Repo: $repoRoot" -ForegroundColor Cyan

# --- 2. Comprobar Docker --------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "!! No se encontro Docker en el PATH de Windows (instala Docker Desktop)." -ForegroundColor Red
    exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! Docker no responde. Esta corriendo Docker Desktop?" -ForegroundColor Red
    exit 1
}

$composeFiles = @("-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")

# --- 3. Construir la imagen GPU (cu128) una sola vez ----------------------
Write-Host "== Construyendo imagen GPU (torch cu128)..." -ForegroundColor Cyan
docker compose @composeFiles build trainer
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! Falle el build de la imagen GPU." -ForegroundColor Red
    exit 1
}

# --- 4. Bucle de rondas ---------------------------------------------------
$ppl = $PPL
$seed = $SEED
$init = $CANON_INIT
$round = 1
$canonPth = "$OUTDIR/puzzle_canon.pth"
$reportHost = "$OUTDIR/puzzle_canon_report.json"

while ($round -le $ROUNDS) {
    $maxg = $ppl * 5

    Write-Host "==== Ronda $round/$ROUNDS | arch=$ARCH | device=$Device | puzzles/nivel=$ppl | max-games=$maxg | seed=$seed | eval-seed=$EVAL_SEED | init=$init | run=retry$round ====" -ForegroundColor Cyan

    $train = @(
        "python", "-m", "trainPuzzles",
        "--arch", $ARCH,
        "--device", $Device,
        "--levels", "1", "--max-level", "$MAX_LEVEL", "--threshold", (InvFmt $THRESHOLD),
        "--start-level", "$START_LEVEL", "--stop-at-level", "$TARGET_LEVEL",
        "--games-budget", "0", "--max-games", "$maxg",
        "--puzzles-per-level", "$ppl",
        "--dropout", (InvFmt $DROPOUT), "--weight-decay", $WEIGHT_DECAY,
        "--lr-schedule", $LR_SCHEDULE, "--lr-final-frac", (InvFmt $LR_FINAL_FRAC),
        "--eval-every", "$EVAL_EVERY",
        "--mine-every", "$MINE_EVERY", "--mine-boost", (InvFmt $MINE_BOOST), "--mine-warmup", "$MINE_WARMUP",
        "--ema-decay", (InvFmt $EMA_DECAY),
        "--rank-coef", (InvFmt $RANK_COEF), "--rank-margin", (InvFmt $RANK_MARGIN),
        "--max-epochs", "$MAX_EPOCHS", "--plateau-stop", "$PLATEAU",
        "--seed", "$seed", "--eval-seed", "$EVAL_SEED",
        "--init", $init, "--out", $canonPth, "--run-name", "retry$round"
    )
    if ($TTA -eq 1) { $train += "--tta" }
    if ($HARD_MINING -eq 1) { $train += "--hard-mining" }
    if ($EMA -eq 1) { $train += "--ema" }

    docker compose @composeFiles run --rm --no-deps -v ./trainPuzzles:/app/trainPuzzles trainer @train
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! Falle docker compose run en la ronda $round." -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path $reportHost)) {
        Write-Host "!! No se encontro el reporte $reportHost; abortando." -ForegroundColor Red
        exit 1
    }

    $report = Get-Content -Path $reportHost -Raw | ConvertFrom-Json
    $target = $report.levels | Where-Object { $_.level -eq $TARGET_LEVEL }
    $mastered = "0"
    if ($target -and $target.mastered -eq $true) { $mastered = "1" }

    Write-Host "---- Ronda $round : mate $TARGET_LEVEL mastered=$mastered"

    if ($mastered -eq "1") {
        Write-Host "**** mate $TARGET_LEVEL DOMINADO en la ronda $round. Modelo canon: $canonPth" -ForegroundColor Green
        exit 0
    }

    $ppl += $PPL_STEP
    $seed += 1
    if ($WARM_START -eq 1) { $init = $canonPth }
    $round += 1
}

Write-Host "**** Se agotaron las $ROUNDS rondas sin dominar mate $TARGET_LEVEL." -ForegroundColor Yellow
exit 2
