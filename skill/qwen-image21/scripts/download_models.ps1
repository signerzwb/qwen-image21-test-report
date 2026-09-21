# Download Qwen-Image-2.1 + both PE models from ModelScope into ./models.
# Total: image ~33GB, each PE ~19GB. Resumable (modelscope cache markers).
$ErrorActionPreference = "Stop"
$py = "$env:TEMP\qwen21_download.py"
@"
from modelscope import snapshot_download
for name in ["Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1-PE-T2I", "Qwen/Qwen-Image-2.1-PE-I2I"]:
    p = snapshot_download(name, local_dir="models/" + name.split("/")[-1])
    print("DOWNLOAD_DONE", p)
"@ | Set-Content -Encoding UTF8 -Path $py
& .\venv\Scripts\python.exe $py
