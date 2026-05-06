$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($conn) {
    $procId = $conn.OwningProcess
    Write-Host "Killing PID $procId on port 8000"
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}
Set-Location "c:\Users\Pks\Downloads\RAG Avant OCR"
Start-Process python -ArgumentList "-m uvicorn api:app --host 127.0.0.1 --port 8000" -WindowStyle Hidden
Write-Host "Server restarted"
