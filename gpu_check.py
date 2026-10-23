import shutil
import subprocess

import torch


print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled in: {torch.backends.cuda.is_built()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"nvidia-smi found: {nvidia_smi is not None}")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"GPU query exit code: {result.returncode}")
        print(result.stdout.strip() or result.stderr.strip())
