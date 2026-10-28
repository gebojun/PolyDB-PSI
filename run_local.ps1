$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

python scripts/run_all.py --device auto --min-free-gpu-gb 2.0
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
