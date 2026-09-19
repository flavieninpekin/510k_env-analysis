param([string]$Mode = "dynamic", [int]$Seed = 0, [int]$Total = 1000000, [int]$Chunk = 50000)

Set-Location (Split-Path -Parent $PSScriptRoot)
for ($i = 0; $i -lt 120; $i++) {
    python -m baselines.train_mappo --mode $Mode --seed $Seed --total $Total --chunk $Chunk --out runs
    if (Test-Path "runs\mappo_${Mode}_s${Seed}.done") { break }
}
