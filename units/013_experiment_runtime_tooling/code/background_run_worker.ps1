param(
    [Parameter(Mandatory = $true)][string]$StatusPath,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [Parameter(Mandatory = $true)][string]$ScheduledStopUtc,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs
)

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

$status = Get-Content -LiteralPath $StatusPath | ConvertFrom-Json
try {
    $status.status = "running"
    $status.worker_pid = $PID
    $status.actual_start_utc = [DateTime]::UtcNow.ToString("o")
    Write-Status $StatusPath $status

    $deadlineUtc = [DateTime]::Parse($ScheduledStopUtc).ToUniversalTime()
    $timeoutSeconds = [Math]::Max(1, [int][Math]::Ceiling(($deadlineUtc - [DateTime]::UtcNow).TotalSeconds))

    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $PythonArgs `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru `
        -WindowStyle Hidden

    $status.python_pid = $proc.Id
    Write-Status $StatusPath $status

    $finished = Wait-Process -Id $proc.Id -Timeout $timeoutSeconds -ErrorAction SilentlyContinue
    if (-not $finished) {
        $proc.Refresh()
        if (-not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
        }
        $status.status = "stopped"
        $status.stop_reason = "deadline"
        $status.timeout_hit = $true
        $status.exit_code = $null
    } else {
        $proc.Refresh()
        $status.exit_code = $proc.ExitCode
        if ($proc.ExitCode -eq 0) {
            $status.status = "completed"
        } else {
            $status.status = "failed"
            $status.stop_reason = "nonzero_exit"
        }
    }
} catch {
    $status.status = "failed"
    $status.stop_reason = "worker_exception"
    $status.error = $_ | Out-String
} finally {
    $status.end_utc = [DateTime]::UtcNow.ToString("o")
    Write-Status $StatusPath $status
}
