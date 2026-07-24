"""AIRT-Engine Configuration"""
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')

@dataclass
class AIRTConfig:
    """Main configuration for AIRT-Engine"""
    
    # Model configuration
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    model_type: str = "text"  # text, vision, audio
    model_cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "airt", "models")
    
    # Inference optimization
    backend: str = "auto"  # auto, llama_cpp, vllm, transformers
    device: str = "auto"   # auto, cpu, cuda
    dtype: str = "auto"    # auto, fp16, fp32, int8, int4
    
    # Quantization
    quantize: bool = True
    quantization_method: str = "gguf"  # gguf, gptq, awq, bitsandbytes
    quantization_bits: int = 4  # 2, 3, 4, 8
    
    # KV Cache optimization
    kv_cache_optimize: bool = True
    kv_cache_method: str = "h2o"  # h2o, streamingllm, kvquant
    kv_cache_size: int = 2048  # tokens to keep
    
    # Speculative decoding
    use_speculative: bool = False
    draft_model: Optional[str] = None
    
    # Memory
    max_seq_len: int = 2048
    gpu_memory_utilization: float = 0.9
    cpu_threads: int = 4
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Multi-modal
    vision_model: str = "vikhyatk/moondream2"
    stt_model: str = "openai/whisper-tiny"
    tts_engine: str = "edge-tts"
    
    # Logging
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> 'AIRTConfig':
        path = path or CONFIG_PATH
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

# Global config
config = AIRTConfig()