# =====================================================================
# push_to_github.ps1
# Sube el contenido de "lab A" a https://github.com/AlbertoAguila/mlops-aap
# en una nueva rama llamada "lab A".
#
# Uso (desde PowerShell, no desde el bash de Claude Code):
#     cd "C:\Users\alber\OneDrive\Desktop\mlops\lab A"
#     .\push_to_github.ps1
#
# Si la primera vez te pide credenciales, se abrirá una ventana del
# navegador para autenticar con GitHub (Git Credential Manager).
# =====================================================================

$ErrorActionPreference = "Stop"

$repoPath  = "C:\Users\alber\OneDrive\Desktop\mlops\lab A"
$remoteUrl = "https://github.com/AlbertoAguila/mlops-aap.git"
$branch    = "lab A"
$message   = "Lab A: Titanic drift analysis with EvidentlyAI"

function Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }

Set-Location -LiteralPath $repoPath
Write-Host "Carpeta: $(Get-Location)"

# ---------------------------------------------------------------------
# 1. Comprobar git
# ---------------------------------------------------------------------
Step 1 "Comprobando que git está instalado..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git no está en el PATH. Instálalo desde https://git-scm.com/download/win y reinicia PowerShell."
    exit 1
}
git --version

# ---------------------------------------------------------------------
# 2. Inicializar repo si hace falta
# ---------------------------------------------------------------------
Step 2 "Asegurando repositorio git..."
if (-not (Test-Path ".git")) {
    git init | Out-Null
    Write-Host "  -> repositorio nuevo creado"
} else {
    Write-Host "  -> ya existía un repositorio"
}

# ---------------------------------------------------------------------
# 3. Identidad de git (sólo si no está configurada)
# ---------------------------------------------------------------------
Step 3 "Configurando identidad..."
if (-not (git config user.name))  { git config user.name  "Alberto" }
if (-not (git config user.email)) { git config user.email "albertoaguila2003@gmail.com" }
Write-Host "  -> $(git config user.name) <$(git config user.email)>"

# ---------------------------------------------------------------------
# 4. Configurar remote origin (lo crea o lo actualiza)
# ---------------------------------------------------------------------
Step 4 "Configurando remote origin..."
$hasOrigin = (git remote) -split "`n" | Where-Object { $_ -eq "origin" }
if ($hasOrigin) {
    git remote set-url origin $remoteUrl
    Write-Host "  -> origin actualizado"
} else {
    git remote add origin $remoteUrl
    Write-Host "  -> origin añadido"
}
Write-Host "  -> $(git remote get-url origin)"

# ---------------------------------------------------------------------
# 5. Cambiar / crear la rama
# ---------------------------------------------------------------------
Step 5 "Cambiando a la rama '$branch'..."
git show-ref --verify --quiet "refs/heads/$branch"
if ($LASTEXITCODE -eq 0) {
    git checkout $branch | Out-Null
    Write-Host "  -> rama existente, ahora estamos en ella"
} else {
    git checkout -b $branch | Out-Null
    Write-Host "  -> rama '$branch' creada"
}
Write-Host "  -> rama actual: $(git rev-parse --abbrev-ref HEAD)"

# ---------------------------------------------------------------------
# 6. Añadir y comitear
# ---------------------------------------------------------------------
Step 6 "Añadiendo cambios..."
git add -A
$pending = git status --porcelain
if ($pending) {
    git commit -m $message | Out-Null
    Write-Host "  -> commit hecho"
} else {
    Write-Host "  -> nada nuevo que comitear"
}

# ---------------------------------------------------------------------
# 7. Push
# ---------------------------------------------------------------------
Step 7 "Subiendo a GitHub..."
Write-Host "  (si es la primera vez, se abrirá el navegador para autenticar)"
git push -u origin $branch

Write-Host ""
Write-Host "============================================================"
Write-Host "OK. Rama subida:" -ForegroundColor Green
Write-Host "    https://github.com/AlbertoAguila/mlops-aap/tree/lab%20A"
Write-Host "============================================================"
