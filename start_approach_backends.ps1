$ErrorActionPreference = "Stop"

$services = @(
    @{ Name = "Approach A"; Path = "C:\Users\Pks\Downloads\RAG Avant OCR"; Port = 8000 },
    @{ Name = "Approach B"; Path = "C:\Users\Pks\Downloads\original(khdam baqi a LLM d 2 table)"; Port = 8001 },
    @{ Name = "Approach C"; Path = "C:\Users\Pks\Downloads\RAG - Copie"; Port = 8002 }
)

foreach ($service in $services) {
    Write-Host "Starting $($service.Name) on port $($service.Port)..."
    Start-Process `
        -FilePath "python" `
        -ArgumentList @("-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$($service.Port)") `
        -WorkingDirectory $service.Path `
        -WindowStyle Hidden
}

Write-Host "Approach backends started: A=http://127.0.0.1:8000 B=http://127.0.0.1:8001 C=http://127.0.0.1:8002"
