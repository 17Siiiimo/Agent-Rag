$ErrorActionPreference = "Stop"

# Kill all python processes that listen on 8000
$procs = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
foreach ($proc in $procs) {
    $procId = $proc.OwningProcess
    Write-Host "Killing PID $procId"
    & cmd /c "taskkill /F /T /PID $procId"
    Start-Sleep -Seconds 1
}

Start-Sleep -Seconds 2

# Verify port is free
$check = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($check) {
    Write-Host "Port still in use by PID $($check.OwningProcess) - trying again"
    & cmd /c "taskkill /F /T /PID $($check.OwningProcess)"
    Start-Sleep -Seconds 2
}

# Start new server
Set-Location "c:\Users\Pks\Downloads\RAG Avant OCR"
$proc = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000" -PassThru -WindowStyle Normal
Write-Host "New server started PID=$($proc.Id)"
