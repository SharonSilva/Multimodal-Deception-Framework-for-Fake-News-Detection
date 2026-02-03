"""
Memory optimization utilities for Railway deployment
"""
import torch
import gc
import os

def optimize_torch_memory():
    """Configure PyTorch for minimal memory usage"""
    # Limit CPU threads
    torch.set_num_threads(1)
    
    # Disable gradient computation (inference only)
    torch.set_grad_enabled(False)
    
    # Use CPU (already set, but making it explicit)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("✅ PyTorch memory optimizations applied")

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()