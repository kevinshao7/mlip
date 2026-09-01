param(
    [string]$OrcaCommand = $env:ORCA_COMMAND
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OrcaCommand)) {
    $OrcaCommand = "orca"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$atoms = @("H", "O", "N", "S")

$orca = Get-Command $OrcaCommand -ErrorAction SilentlyContinue
if ($null -eq $orca) {
    throw "ORCA command is not available: $OrcaCommand"
}

foreach ($atom in $atoms) {
    $runDir = Join-Path $scriptDir "runs\$atom"
    $stem = "orcaatomization$atom"
    $inputPath = Join-Path $runDir "$stem.inp"
    $outputPath = Join-Path $runDir "$stem.out"
    $basisPath = Join-Path $runDir "def2-tzvpd.bas"

    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Missing ORCA input: $inputPath"
    }
    if (-not (Test-Path -LiteralPath $basisPath -PathType Leaf)) {
        throw "Missing basis file in run directory: $basisPath"
    }

    Write-Host "Running $stem with ORCA command: $($orca.Source)"
    Write-Host "Input: $inputPath"
    Write-Host "Output: $outputPath"

    Push-Location $runDir
    try {
        & $orca.Source "$stem.inp" 2>&1 | Tee-Object -FilePath "$stem.out"
        if ($LASTEXITCODE -ne 0) {
            throw "ORCA exited with code $LASTEXITCODE for $stem"
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Select-String -Path $outputPath -Pattern "FINAL SINGLE POINT ENERGY" -Quiet)) {
        throw "Missing FINAL SINGLE POINT ENERGY in $outputPath"
    }
    if (-not (Select-String -Path $outputPath -Pattern "ORCA TERMINATED NORMALLY" -Quiet)) {
        throw "Missing ORCA TERMINATED NORMALLY in $outputPath"
    }

    Write-Host "Finished $stem"
}
