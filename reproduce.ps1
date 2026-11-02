param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$sourceRoot = $PSScriptRoot
$runRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
if (Test-Path -LiteralPath $runRoot) { throw "Use a new output directory to preserve the recorded run: $runRoot" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
foreach ($fileName in @('run_project.py','config.json','PROTOCOL.md','verify_results.py','analyse_results.py','audit_selected_examples.py')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $fileName) -Destination $runRoot
}
Push-Location $runRoot
try {
    foreach ($stage in @('prepare','features','train','evaluate')) {
        python -u run_project.py $stage
        if ($LASTEXITCODE -ne 0) { throw "Stage $stage failed. Keep logs and repair in a new run directory." }
    }
    python verify_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Independent verification failed' }
    python analyse_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
    python audit_selected_examples.py
    if ($LASTEXITCODE -ne 0) { throw 'Selected-example token audit failed' }
} finally { Pop-Location }
