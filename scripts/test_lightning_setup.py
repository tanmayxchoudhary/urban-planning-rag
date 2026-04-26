#!/usr/bin/env python3
"""
Test Lightning AI Setup - Verify all imports work
"""

import sys

def test_imports():
    """Test all critical imports"""
    tests = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("PIL", "PIL"),
        ("fitz (PyMuPDF)", "fitz"),
        ("chromadb", "chromadb"),
        ("colpali_engine", "colpali_engine.models"),
        ("transformers", "transformers"),
    ]
    
    print("🔍 Testing imports...")
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed imports: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ All imports successful!")
        
    # Test CUDA
    import torch
    if torch.cuda.is_available():
        print(f"🖥️  CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    # Test NumPy version
    import numpy as np
    print(f"📊 NumPy version: {np.__version__}")
    
    return True

if __name__ == "__main__":
    test_imports()
