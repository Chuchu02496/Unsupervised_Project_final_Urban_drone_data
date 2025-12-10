@echo off
echo Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.
echo Verifying CUDA installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')"
pause
