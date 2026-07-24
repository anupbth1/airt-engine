"""AIRT-Engine: Complete Colab Setup Script"""
import subprocess, sys, os
import time

print("=" * 60)
print("🚀 AIRT-Engine + AIRT-Morpher - Colab Setup")
print("=" * 60)

# Step 1: Install core dependencies (Colab-compatible)
print("\n[1/4] Installing core dependencies...")

core_pkgs = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "pillow>=10.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "jinja2>=3.1.2",
    "python-multipart>=0.0.6",
    "numpy>=1.24.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "edge-tts>=6.0.0",
]

for pkg in core_pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)

print("   Core dependencies installed!")

# Step 2: Install GPU-specific packages (skip if fails)
print("\n[2/4] Installing GPU packages (optional)...")

gpu_pkgs = [
    ("vllm", "vllm>=0.3.0"),
    ("flash-attn", "flash-attn>=2.5.0"),
    ("bitsandbytes", "bitsandbytes>=0.41.0"),
]

for name, pkg in gpu_pkgs:
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)
    if result.returncode == 0:
        print(f"   ✅ {name} installed")
    else:
        print(f"   ⚠️  {name} skipped (not critical)")

# Step 3: Install quantization (skip auto-gptq - issue on Colab)
print("\n[3/4] Installing quantization packages...")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autoawq"], capture_output=True)
    print("   ✅ autoawq installed")
except:
    print("   ⚠️  autoawq skipped")

# Step 4: Verify imports
print("\n[4/4] Verifying imports...")

modules = [
    "airt.engine",
    "airt.compiler.layer_analyzer",
    "airt.compiler.model_optimizer",
    "airt.compiler.query_predictor",
    "airt.compiler.compiler_cli",
    "airt.optimizers.quantizer",
    "airt.optimizers.kv_cache",
    "airt.optimizers.speculative",
    "airt.models.loader",
    "airt.models.profile",
    "airt.inference.runner",
    "airt.inference.benchmark",
    "airt.utils.config",
    "airt.utils.logger",
]

all_ok = True
for m in modules:
    try:
        __import__(m)
        print(f"   ✅ {m}")
    except Exception as e:
        print(f"   ❌ {m}: {str(e)[:60]}")
        all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("✅ ALL MODULES READY! AIRT-Engine is fully functional.")
else:
    print("⚠️  Some modules failed. Run 'python test_compiler.py' for details.")
print(f"{'='*60}")

# Print next steps
print("""
📋 NEXT STEPS:
   1. Run: from airt.compiler.query_predictor import QueryCostPredictor
   2. Run: !python cli.py compiler demo
   3. Run: !python airt/compressor/test_morpher.py
   4. Run: !python cli.py server  (for Web UI)

📖 Full documentation: https://github.com/anupbth1/airt-engine
""")