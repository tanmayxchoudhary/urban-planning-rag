#!/usr/bin/env python3
"""Quick validation test for embed_v29.py - runs on Lightning AI"""

import sys
import time

print("="*60)
print("VALIDATION TEST - embed_v29.py")
print("="*60)

# Test 1: Check imports
print("\n[1/4] Checking imports...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
    
    from transformers import AutoModel, AutoProcessor
    print("  ✓ transformers")
    
    import fitz
    print("  ✓ PyMuPDF")
    
    from PIL import Image
    print("  ✓ Pillow")
    
    import numpy as np
    print("  ✓ numpy")
    
except ImportError as e:
    print(f"  ✗ Missing: {e}")
    sys.exit(1)

# Test 2: Check GPU
print("\n[2/4] Checking GPU...")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ GPU: {gpu}")
    print(f"  ✓ VRAM: {vram:.1f} GB")
else:
    print("  ✗ No GPU - will be very slow")

# Test 3: Create test PDF
print("\n[3/4] Creating test PDF...")
doc = fitz.open()
for i in range(10):
    page = doc.new_page(612, 792)
    page.insert_text((72, 72), f"Test Page {i+1}\n\nContent here", fontsize=12)
doc.save("/tmp/test10.pdf")
doc.close()
print("  ✓ Created 10-page test PDF")

# Test 4: Load model and embed
print("\n[4/4] Testing model load + embed...")
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"

try:
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        max_num_visual_tokens=1280
    )
    print(f"  ✓ Processor loaded ({time.time()-t0:.1f}s)")
    
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    ).eval()
    print(f"  ✓ Model loaded ({time.time()-t0:.1f}s)")
    
    # Test torch.compile
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  ✓ Model compiled with torch.compile()")
        except Exception as e:
            print(f"  ! Compile skipped: {e}")
    
    # Embed 1 page
    doc = fitz.open("/tmp/test10.pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    t0 = time.time()
    features = processor.process_images([img])
    features = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in features.items()}
    
    with torch.no_grad():
        out = model(**features)
        emb = out.embeddings[0].cpu()
    
    elapsed = time.time() - t0
    print(f"  ✓ Embedded 1 page in {elapsed:.2f}s")
    print(f"  ✓ Embedding shape: {emb.shape}")
    
    # Estimate for 3000 pages
    est_3000 = elapsed * 3000 / 60
    print(f"\n  📊 ESTIMATE: 3000 pages = {est_3000:.1f} minutes")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
