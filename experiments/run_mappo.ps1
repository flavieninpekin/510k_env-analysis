# Launch MAPPO self-play runs (DYNAMIC, seeds 0-2) as crash-tolerant chunk loops.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_mappo.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
"start $(Get-Date -Format o)" | Out-File runs\mappo_status.txt

$procs = @()
foreach ($s in 0, 1, 2) {
    $procs += Start-Process -FilePath 'powershell' `
        -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                      'experiments\mappo_seed.ps1', '-Mode', 'dynamic', '-Seed', "$s" `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput "runs\mappo_dynamic_s$s.log" `
        -RedirectStandardError "runs\mappo_dynamic_s$s.err.log" `
        -WindowStyle Hidden -PassThru
}
"launched $($procs.Count) seeds $(Get-Date -Format o)" | Out-File runs\mappo_status.txt -Append
foreach ($p in $procs) { $p.WaitForExit() }
"all done $(Get-Date -Format o)" | Out-File runs\mappo_status.txt -Append
