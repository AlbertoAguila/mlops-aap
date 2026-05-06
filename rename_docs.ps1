# =====================================================================
# rename_docs.ps1
# Renombra entrega/docs/ -> entrega/resultados/ y sube los cambios a
# la rama lab-A en GitHub. Las referencias internas (script, README,
# discusion, .gitignore) ya están actualizadas.
# =====================================================================

$ErrorActionPreference = "Stop"
$repoPath = "C:\Users\alber\OneDrive\Desktop\mlops\lab A"
Set-Location -LiteralPath $repoPath

Write-Host "[1] Renombrando carpeta con git mv (preserva historial) ..." -ForegroundColor Cyan
if (Test-Path "entrega/docs") {
    git mv entrega/docs entrega/resultados
    Write-Host "  -> entrega/docs -> entrega/resultados"
} elseif (Test-Path "entrega/resultados") {
    Write-Host "  -> ya estaba renombrada"
} else {
    Write-Error "No encuentro ni entrega/docs ni entrega/resultados"
    exit 1
}

Write-Host ""
Write-Host "[2] Mostrando cambios pendientes ..." -ForegroundColor Cyan
git status --short

Write-Host ""
Write-Host "[3] Add + commit ..." -ForegroundColor Cyan
git add -A
git commit -m "Rename entrega/docs -> entrega/resultados"

Write-Host ""
Write-Host "[4] Push a origin/lab-A ..." -ForegroundColor Cyan
git push

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "OK. Cambios subidos:"
Write-Host "    https://github.com/AlbertoAguila/mlops-aap/tree/lab-A/entrega/resultados"
Write-Host "============================================================"
