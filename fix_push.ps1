# =====================================================================
# fix_push.ps1
# Repara el intento anterior:
#  - el commit ya existe en master/main (porque "lab A" con espacio
#    no es nombre válido de rama en git)
#  - renombra esa rama a "lab-A" y la sube
# =====================================================================

$ErrorActionPreference = "Stop"
$repoPath = "C:\Users\alber\OneDrive\Desktop\mlops\lab A"
$branch   = "lab-A"

Set-Location -LiteralPath $repoPath

Write-Host "[1] Estado actual:" -ForegroundColor Cyan
git status --short
Write-Host ""
Write-Host "Rama actual: $(git rev-parse --abbrev-ref HEAD)"
Write-Host ""

Write-Host "[2] Renombrando la rama actual a '$branch' ..." -ForegroundColor Cyan
git branch -M $branch
Write-Host "  -> ahora estamos en: $(git rev-parse --abbrev-ref HEAD)"

Write-Host ""
Write-Host "[3] Push a origin/$branch ..." -ForegroundColor Cyan
Write-Host "  (la primera vez se abrirá el navegador para autenticar con GitHub)"
git push -u origin $branch

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "OK. Rama subida:"
Write-Host "    https://github.com/AlbertoAguila/mlops-aap/tree/$branch"
Write-Host "============================================================"
