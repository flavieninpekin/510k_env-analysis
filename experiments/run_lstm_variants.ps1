# LSTM collapse investigation: two hyperparameter variants x 4 modes x 4 seeds.
#
# Trains crash-tolerant run_matrix workers in parallel (8 at a time, one per
# variant x mode, each doing seeds 0-3 sequentially), then runs the
# cooperation-inference probe on every variant checkpoint and aggregates.
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_lstm_variants.ps1
# Status: runs/lstm_variants_status.txt   Logs: runs/lstmvar_*.log

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)

$modes = @('single', 'static', 'dynamic', 'obvious')
$variants = @(
    @{ suffix = 'e05'; extra = @('--ent-coef', '0.05') },
    @{ suffix = 'small'; extra = @('--lr', '0.0001', '--net-arch', '128', '128', '--lstm-hidden', '128') }
)

"start $(Get-Date -Format o)" | Out-File runs\lstm_variants_status.txt

$procs = @()
foreach ($v in $variants) {
    foreach ($m in $modes) {
        $argList = @(
            '-m', 'baselines.run_matrix', '--modes', $m, '--policy', 'lstm',
            '--seeds', '0', '1', '2', '3', '--total', '1000000', '--chunk', '100000',
            '--out', 'runs', '--tag-suffix', $v.suffix
        ) + $v.extra
        $log = "runs\lstmvar_$($v.suffix)_$($m).log"
        $err = "runs\lstmvar_$($v.suffix)_$($m).err.log"
        $procs += Start-Process -FilePath 'python' -ArgumentList $argList `
            -WorkingDirectory (Get-Location).Path `
            -RedirectStandardOutput $log -RedirectStandardError $err `
            -WindowStyle Hidden -PassThru
    }
}
"launched $($procs.Count) workers $(Get-Date -Format o)" | Out-File runs\lstm_variants_status.txt -Append

foreach ($p in $procs) { $p.WaitForExit() }
"training done $(Get-Date -Format o)" | Out-File runs\lstm_variants_status.txt -Append

foreach ($v in $variants) {
    python -m baselines.analyze_team_conditioning `
        --policy lstm --tag-suffix $v.suffix `
        --modes single static dynamic obvious --seeds 0 1 2 3 --n-episodes 500 `
        --out runs *>> "runs\lstmvar_eval_$($v.suffix).log"
}
python -m baselines.aggregate --out runs *>> runs\lstmvar_eval_aggregate.log
"all done $(Get-Date -Format o)" | Out-File runs\lstm_variants_status.txt -Append
