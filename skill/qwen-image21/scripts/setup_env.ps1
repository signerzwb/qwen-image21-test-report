# Qwen-Image-2.1 environment setup (Windows PowerShell).
# Creates ./venv in the current directory. Tested versions pinned below.
$ErrorActionPreference = "Stop"

python -m venv venv
& .\venv\Scripts\python.exe -m pip install -U pip
& .\venv\Scripts\python.exe -m pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& .\venv\Scripts\python.exe -m pip install "transformers==5.17.0" "accelerate==1.15.0" "modelscope==1.40.1" json-repair
& .\venv\Scripts\python.exe -m pip install flash-linear-attention triton-windows

# diffusers ships QwenImage21Pipeline only on main (0.41.0.dev0).
# NEVER git clone (stalls on Windows); use the codeload tarball:
$tar = "$env:TEMP\diffusers-main.tar.gz"
Invoke-WebRequest -Uri "https://codeload.github.com/huggingface/diffusers/tar.gz/refs/heads/main" -OutFile $tar
& .\venv\Scripts\python.exe -m pip install --no-build-isolation $tar

& .\venv\Scripts\python.exe -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"

