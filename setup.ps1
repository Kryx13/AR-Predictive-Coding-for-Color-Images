# ================================================================================================
# Script PowerShell - Création Architecture Projet 8: Color Image Predictive Coding
# Pour PyCharm - Création des dossiers uniquement
# ================================================================================================

param(
    [string]$ProjectName = "project8-predictive-coding"
)

# Couleurs pour les messages
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "📁 $msg" -ForegroundColor Cyan }
function Write-Header { param($msg) Write-Host "`n🎯 $msg" -ForegroundColor Magenta }

Write-Header "Création de l'architecture du projet: $ProjectName"

# Création du dossier racine
if (-not (Test-Path $ProjectName)) {
    New-Item -ItemType Directory -Path $ProjectName -Force | Out-Null
    Write-Success "Dossier racine créé: $ProjectName"
}

# Changement vers le dossier du projet
Set-Location $ProjectName

# Structure des dossiers
$directories = @(
    "src",
    "src\core", 
    "src\utils",
    "src\algorithms",
    "tests",
    "tests\unit",
    "tests\integration",
    "notebooks",
    "data",
    "data\images",
    "data\images\test",
    "data\images\results", 
    "data\metrics",
    "docs",
    "configs",
    "scripts"
)

Write-Info "Création de la structure des dossiers..."

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  📂 $dir" -ForegroundColor Gray
    }
}

# Création des fichiers __init__.py pour les modules Python
$initFiles = @(
    "src\__init__.py",
    "src\core\__init__.py",
    "src\utils\__init__.py", 
    "src\algorithms\__init__.py",
    "tests\__init__.py",
    "tests\unit\__init__.py",
    "tests\integration\__init__.py"
)

Write-Info "Création des fichiers __init__.py..."

foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "  📄 $file" -ForegroundColor Gray
    }
}

# Création d'un .gitignore de base
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
venv/
env/
ENV/

# PyCharm
.idea/
*.iml

# Jupyter
.ipynb_checkpoints/

# Data files
data/results/
*.png
*.jpg
*.jpeg
*.tiff
*.bmp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"@

$gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
Write-Success "Fichier .gitignore créé"

Write-Header "Architecture créée avec succès!"
Write-Host ""
Write-Host "📋 Structure créée:" -ForegroundColor Yellow
Write-Host "  📁 $ProjectName/" -ForegroundColor White
foreach ($dir in $directories) {
    Write-Host "    📁 $dir/" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🚀 Prochaines étapes:" -ForegroundColor Yellow
Write-Host "  1. Ouvrir le projet dans PyCharm" -ForegroundColor White
Write-Host "  2. Configurer l'environnement virtuel Python" -ForegroundColor White
Write-Host "  3. Installer les dépendances: pip install opencv-python numpy scipy matplotlib" -ForegroundColor White

Write-Host ""
Write-Success "Projet prêt pour PyCharm!"