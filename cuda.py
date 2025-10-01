# Diagnostic script - run this separately
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test CUDA
    try:
        x = torch.tensor([1.0]).cuda()
        print("Simple CUDA test: PASSED")
        x = x * 2
        print(f"Computation test: {x.item()} (should be 2.0)")
    except Exception as e:
        print(f"CUDA test FAILED: {e}")