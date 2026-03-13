"""
Check if PyTorch can use your GPU (CUDA). If not, prints the exact pip command to install PyTorch with CUDA.
Run: python check_gpu.py
"""
import sys

def main():
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install base deps first:")
        print("  pip install -r requirements.txt")
        print("Then install PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return 1

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch version: {torch.__version__}")
        return 0

    print("CUDA is not available (PyTorch is using CPU).")
    print("\nTo use your GPU (e.g. RTX 4060), install PyTorch with CUDA support:")
    print("\n  pip uninstall torch torchvision torchaudio -y")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nThen run this script again to verify.")
    print("\nIf cu121 fails, try CUDA 11.8:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    return 1

if __name__ == "__main__":
    sys.exit(main())
