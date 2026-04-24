param(
    [Parameter(Mandatory = $true)][string]$UnitDir,
    [int]$TailLines = 20
)

$resolvedUnitDir = (Resolve-Path -LiteralPath $UnitDir).Path
$statusPath = Join-Path $resolvedUnitDir "run_status.json"
if (-not (Test-Path -LiteralPath $statusPath)) {
    throw "No run_status.json found under $resolvedUnitDir"
}

$status = Get-Content -LiteralPath $statusPath | ConvertFrom-Json
$result = [ordered]@{
    status = $status
}

if (Test-Path -LiteralPath $status.stdout_path) {
    $result.stdout_tail = Get-Content -LiteralPath $status.stdout_path -Tail $TailLines
}
if (Test-Path -LiteralPath $status.stderr_path) {
    $result.stderr_tail = Get-Content -LiteralPath $status.stderr_path -Tail $TailLines
}

$result | ConvertTo-Json -Depth 8
