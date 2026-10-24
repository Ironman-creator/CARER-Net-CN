[CmdletBinding()]
param(
    [ValidateSet('All', 'Preflight', 'Train', 'Test', 'Summary')]
    [string]$Stage = 'All',
    [string]$ProjectRoot = '',
    [string]$DataRoot = '',
    [string]$OutputRoot = '',
    [ValidateRange(0, 16)]
    [int]$Workers = 2,
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [ValidateSet('All', 'Validation', 'Test')]
    [string]$SummaryPhase = 'All'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Join-Path $PSScriptRoot '..'
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $ProjectRoot 'data'
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $PSScriptRoot 'outputs'
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$script:Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$script:Runner = Join-Path $PSScriptRoot 'component_runner.py'
$script:Summary = Join-Path $PSScriptRoot 'component_summary.py'
$script:Logged = Join-Path $PSScriptRoot 'run_logged.py'
$script:ReturnBundle = Join-Path $PSScriptRoot 'make_return_bundle.py'
$script:Seeds = @(42, 123, 3407)
$script:Arms = @('core', 'no_bound', 'random_init')
$script:CommonArguments = @(
    '--project-root', $ProjectRoot,
    '--data-root', $DataRoot,
    '--output-root', $OutputRoot,
    '--workers', [string]$Workers,
    '--device', $Device
)

foreach ($required in @($script:Python, $script:Runner, $script:Summary, $script:Logged, $script:ReturnBundle)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required. No automatic substitution will be attempted."
    }
}

$logRoot = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$script:ConsoleLog = Join-Path $logRoot ("pipeline_{0}_{1}.log" -f $stamp, $PID)
$script:StatusFile = Join-Path $logRoot 'pipeline_status.txt'
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-RunMessage {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    [System.IO.File]::AppendAllText($script:ConsoleLog, $line + [Environment]::NewLine, $script:Utf8)
    [System.IO.File]::WriteAllText($script:StatusFile, $line + [Environment]::NewLine, $script:Utf8)
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    # run_logged merges child stdout/stderr into its own stdout in real time.
    # Its arguments begin with -u and a script, not a second Python executable.
    & $script:Python -u $script:Logged --log-file $script:ConsoleLog -- @Arguments
    $code = $LASTEXITCODE
    if ($code -in @(130, -1073741510, 3221225786)) {
        throw [System.OperationCanceledException]::new('Interrupted by the user. No remaining runs will be started.')
    }
    if ($code -ne 0) {
        throw "Python stage failed with exit code $code. See $script:ConsoleLog"
    }
}

function Invoke-Runner {
    param([string]$Command, [string[]]$ExtraArguments = @())
    Invoke-Python -Arguments (@('-u', $script:Runner, $Command) + $script:CommonArguments + $ExtraArguments)
}

function Invoke-Summary {
    param([ValidateSet('validation', 'test')][string]$Phase)
    Write-RunMessage "SUMMARY $Phase"
    Invoke-Python -Arguments @('-u', $script:Summary, '--output-root', $OutputRoot, '--phase', $Phase)
}

function Invoke-TrainMatrix {
    $failures = New-Object 'System.Collections.Generic.List[string]'
    foreach ($seed in $script:Seeds) {
        foreach ($arm in $script:Arms) {
            $label = '{0}/seed_{1}' -f $arm, $seed
            Write-RunMessage "TRAIN $label"
            try {
                Invoke-Runner -Command 'train' -ExtraArguments @('--arm', $arm, '--seed', [string]$seed)
                Write-RunMessage "TRAIN COMPLETE $label"
            }
            catch {
                if ($_.Exception -is [System.OperationCanceledException]) { throw }
                $failures.Add($label)
                Write-RunMessage ("TRAIN FAILED {0}: {1}" -f $label, $_.Exception.Message)
            }
        }
    }
    if ($failures.Count -gt 0) {
        throw ('Training failures: {0}. Seal and official tests were NOT started. Rerun the same stage after fixing the cause.' -f ($failures -join ', '))
    }
    Write-RunMessage 'All nine training runs completed. Sealing validation-selected checkpoints.'
    Invoke-Runner -Command 'seal'
    Invoke-Summary -Phase 'validation'
}

function Invoke-TestMatrix {
    $seal = Join-Path $OutputRoot 'training_seal.json'
    if (-not (Test-Path -LiteralPath $seal -PathType Leaf)) {
        throw 'Official tests require an existing all-nine training_seal.json. Run Stage Train successfully first.'
    }
    # The runner verifies the full seal and checkpoint hashes before opening test images.
    $failures = New-Object 'System.Collections.Generic.List[string]'
    foreach ($seed in $script:Seeds) {
        foreach ($arm in $script:Arms) {
            $label = '{0}/seed_{1}' -f $arm, $seed
            Write-RunMessage "TEST $label"
            try {
                Invoke-Runner -Command 'test' -ExtraArguments @('--arm', $arm, '--seed', [string]$seed)
                Write-RunMessage "TEST COMPLETE $label"
            }
            catch {
                if ($_.Exception -is [System.OperationCanceledException]) { throw }
                $failures.Add($label)
                Write-RunMessage ("TEST FAILED {0}: {1}" -f $label, $_.Exception.Message)
            }
        }
    }
    if ($failures.Count -gt 0) {
        throw ('Test failures: {0}. No complete test summary was generated.' -f ($failures -join ', '))
    }
    Invoke-Summary -Phase 'test'
    Write-RunMessage 'Creating a verified return ZIP without large checkpoint files.'
    Invoke-Python -Arguments @('-u', $script:ReturnBundle, '--output-root', $OutputRoot)
}

$oldUtf8 = [Environment]::GetEnvironmentVariable('PYTHONUTF8', 'Process')
$oldCublas = [Environment]::GetEnvironmentVariable('CUBLAS_WORKSPACE_CONFIG', 'Process')
$oldIoEncoding = [Environment]::GetEnvironmentVariable('PYTHONIOENCODING', 'Process')
$pipelineLock = $null
$exitCode = 0
try {
    $lockPath = Join-Path $logRoot 'pipeline.lock'
    $pipelineLock = [System.IO.File]::Open(
        $lockPath, [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None
    )
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8', 'Process')
    [Environment]::SetEnvironmentVariable('CUBLAS_WORKSPACE_CONFIG', ':4096:8', 'Process')
    Write-RunMessage "START stage=$Stage project=$ProjectRoot outputs=$OutputRoot"
    Write-RunMessage "LOG $script:ConsoleLog"
    Invoke-Runner -Command 'check-package'
    switch ($Stage) {
        'All' {
            Invoke-Runner -Command 'preflight'
            Invoke-TrainMatrix
            Invoke-TestMatrix
        }
        'Preflight' {
            Invoke-Runner -Command 'preflight'
        }
        'Train' {
            Invoke-Runner -Command 'preflight'
            Invoke-TrainMatrix
        }
        'Test' {
            Invoke-TestMatrix
        }
        'Summary' {
            if ($SummaryPhase -in @('All', 'Validation')) {
                Invoke-Summary -Phase 'validation'
            }
            if ($SummaryPhase -in @('All', 'Test')) {
                Invoke-Summary -Phase 'test'
            }
        }
    }
    Write-RunMessage "COMPLETE stage=$Stage. Keep outputs including sealed best checkpoints for full verification."
}
catch {
    $exitCode = 1
    Write-RunMessage ("STOPPED: {0}" -f $_.Exception.Message)
}
finally {
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', $oldUtf8, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONIOENCODING', $oldIoEncoding, 'Process')
    [Environment]::SetEnvironmentVariable('CUBLAS_WORKSPACE_CONFIG', $oldCublas, 'Process')
    if ($null -ne $pipelineLock) {
        $pipelineLock.Dispose()
    }
}
exit $exitCode
