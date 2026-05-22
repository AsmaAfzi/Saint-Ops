# Repair Windows ML env after mlflow install (protobuf conflict breaks TensorFlow).
# Run from repo root:  .\ml\fix_env.ps1

Write-Host "Fixing TensorFlow + MLflow dependency pins..."
python -m pip install -r "$PSScriptRoot\requirements.txt" --upgrade

Write-Host "`nVerifying imports..."
python -c "import tensorflow as tf; import mlflow; print('tensorflow', tf.__version__); print('mlflow', mlflow.__version__); import google.protobuf; print('protobuf', google.protobuf.__version__)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nOK — run:  cd ml; python train_model.py"
} else {
    Write-Host "`nIf TensorFlow still fails, install Microsoft VC++ Redistributable:"
    Write-Host "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
}
