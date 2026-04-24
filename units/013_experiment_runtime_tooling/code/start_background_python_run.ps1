param(
    [Parameter(Mandatory = $true)][string]$UnitDir,
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][int]$SliceSeconds,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs
)

function New-Directory($path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Write-Status($path, $status) {
    $json = $status | ConvertTo-Json -Depth 8
    for ($i = 0; $i -lt 10; $i++) {
        try {
            Set-Content -LiteralPath $path -Value $json
            return
        } catch {
            Start-Sleep -Milliseconds 100
        }
    }
    throw "Unable to write status file: $path"
}

$resolvedUnitDir = (Resolve-Path -LiteralPath $UnitDir).Path
$modelRunDir = Join-Path $resolvedUnitDir ("model\\runs\\" + $RunName)
$plotRunDir = Join-Path $resolvedUnitDir ("plot\\runs\\" + $RunName)
$statusPath = Join-Path $resolvedUnitDir "run_status.json"
$stdoutPath = Join-Path $plotRunDir "stdout.log"
$stderrPath = Join-Path $plotRunDir "stderr.log"
$scheduledStopUtc = [DateTime]::UtcNow.AddSeconds($SliceSeconds).ToString("o")

New-Directory $modelRunDir
New-Directory $plotRunDir

$status = [ordered]@{
    unit_dir = $resolvedUnitDir
    run_name = $RunName
    status = "queued"
    command = @($PythonExe) + $PythonArgs
    working_directory = $resolvedUnitDir
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    model_run_dir = $modelRunDir
    plot_run_dir = $plotRunDir
    created_utc = [DateTime]::UtcNow.ToString("o")
    scheduled_stop_utc = $scheduledStopUtc
}
Write-Status $statusPath $status

$workerScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "background_run_worker.ps1")).Path
$workerArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $workerScript,
    "-StatusPath", $statusPath,
    "-PythonExe", $PythonExe,
    "-WorkingDirectory", $resolvedUnitDir,
    "-StdoutPath", $stdoutPath,
    "-StderrPath", $stderrPath,
    "-ScheduledStopUtc", $scheduledStopUtc
) + $PythonArgs

$worker = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $workerArgs `
    -WindowStyle Hidden `
    -PassThru

$status.worker_launcher_pid = $worker.Id
$status
