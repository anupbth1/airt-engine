#!/usr/bin/env python3
"""
AIRT-Engine: Adaptive Intelligence Runtime Engine

A unified multi-modal AI system that combines:
- Text Chat (via vLLM/llama.cpp/Transformers)
- Image Analysis (via MiniCPM-V/LLaVA/Moondream)
- Speech-to-Text (via Whisper)
- Text-to-Speech (via edge-tts)
- Benchmarking + Profiling
- Web UI + CLI + Python API

Quick Start:
  # Web UI
  python main.py server
  
  # Command line
  python main.py chat "What is quantum computing?"
  python main.py image photo.jpg "What is this?"
  python main.py transcribe speech.mp3
  
  # Python
  from airt.engine import AIRTEngine
  engine = AIRTEngine()
  response = engine.chat("Hello!")
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # Run CLI
    from cli import main
    main()