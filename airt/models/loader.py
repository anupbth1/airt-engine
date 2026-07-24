"""AIRT-Engine Model Loader - Unified interface for all model types"""
import os
import torch
from typing import Optional, Any, Callable, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from airt.utils.logger import log
from airt.utils.config import config
from airt.optimizers.quantizer import Quantizer


class ModelLoader:
    """Load models with automatic optimization selection based on hardware."""
    
    def __init__(self):
        self.models = {}  # Cache loaded models
        self.tokenizers = {}
        self.quantizer = Quantizer()
    
    def detect_best_backend(self) -> str:
        """Auto-detect best backend based on available hardware."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
            
            if vram >= 24:
                return "vllm"  # High-end GPU
            elif vram >= 8:
                return "transformers"  # Mid-range GPU
            else:
                return "transformers"
        else:
            log.info("No GPU detected, using CPU backend")
            return "llama_cpp"
    
    def load_text_model(self, model_name: str, **kwargs) -> Tuple[Any, Any]:
        """
        Load a text model with optimal settings.
        
        Args:
            model_name: HuggingFace model name
            **kwargs: Additional loading parameters
            
        Returns:
            (model, tokenizer) tuple
        """
        cache_key = f"text:{model_name}"
        if cache_key in self.models:
            log.info(f"Using cached model: {model_name}")
            return self.models[cache_key], self.tokenizers[cache_key]
        
        backend = kwargs.get('backend', config.backend)
        if backend == 'auto':
            backend = self.detect_best_backend()
        
        log.info(f"Loading text model: {model_name} (backend={backend})")
        
        if backend == 'llama_cpp':
            model, tokenizer = self._load_llamacpp(model_name, **kwargs)
        elif backend == 'vllm':
            model, tokenizer = self._load_vllm(model_name, **kwargs)
        else:
            model, tokenizer = self._load_transformers(model_name, **kwargs)
        
        self.models[cache_key] = model
        self.tokenizers[cache_key] = tokenizer
        
        return model, tokenizer
    
    def _load_llamacpp(self, model_name: str, **kwargs) -> Tuple[Any, Any]:
        """Load model using llama.cpp for CPU-optimized inference."""
        try:
            from llama_cpp import Llama
            
            # Determine GGUF path
            gguf_path = kwargs.get('gguf_path')
            if not gguf_path:
                # Search for GGUF file in cache
                cache_dir = config.model_cache_dir
                gguf_name = model_name.replace('/', '_') + f"_Q{config.quantization_bits}.gguf"
                gguf_path = os.path.join(cache_dir, gguf_name)
                
                if not os.path.exists(gguf_path):
                    log.warning(f"GGUF file not found: {gguf_path}")
                    log.info("Tip: Download GGUF models from HuggingFace or convert using llama.cpp")
                    log.info(f"Example: TheBloke/{model_name.split('/')[-1]}-GGUF")
                    # Try to download from TheBloke
                    bloke_name = f"TheBloke/{model_name.split('/')[-1]}-GGUF"
                    log.info(f"Trying: {bloke_name}")
                    gguf_path = bloke_name
            
            log.info(f"Loading llama.cpp model from: {gguf_path}")
            
            n_ctx = kwargs.get('max_seq_len', config.max_seq_len)
            n_threads = kwargs.get('cpu_threads', config.cpu_threads)
            
            model = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # CPU mode
                verbose=False,
            )
            
            # llama.cpp has built-in tokenizer
            tokenizer = model.tokenizer() if hasattr(model, 'tokenizer') else None
            
            log.info("llama.cpp model loaded successfully")
            return model, tokenizer
            
        except ImportError:
            log.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            log.info("Fallback: Using HuggingFace transformers")
            return self._load_transformers(model_name, **kwargs)
        except Exception as e:
            log.warning(f"llama.cpp loading failed: {e}")
            log.info("Fallback: Using HuggingFace transformers")
            return self._load_transformers(model_name, **kwargs)
    
    def _load_vllm(self, model_name: str, **kwargs) -> Tuple[Any, Any]:
        """Load model using vLLM for GPU-optimized inference."""
        try:
            from vllm import LLM, SamplingParams
            
            log.info(f"Loading vLLM model: {model_name}")
            
            quantization = kwargs.get('quantization')
            if not quantization:
                # Auto-detect quantization from model name
                name_lower = model_name.lower()
                if 'awq' in name_lower:
                    quantization = 'awq'
                elif 'gptq' in name_lower:
                    quantization = 'gptq'
                elif config.quantize:
                    quantization = 'awq'  # Default
            
            vllm_kwargs = {
                'model': model_name,
                'trust_remote_code': True,
                'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
                'gpu_memory_utilization': kwargs.get('gpu_memory_utilization', config.gpu_memory_utilization),
                'seed': kwargs.get('seed', 42),
            }
            
            if quantization:
                vllm_kwargs['quantization'] = quantization
            
            model = LLM(**vllm_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            log.info(f"vLLM model loaded: {model_name}")
            return model, tokenizer
            
        except ImportError:
            log.error("vllm not installed. Run: pip install vllm")
            raise
    
    def _load_transformers(self, model_name: str, **kwargs) -> Tuple[Any, Any]:
        """Load model using HuggingFace transformers."""
        log.info(f"Loading transformers model: {model_name}")
        
        dtype_map = {
            'fp16': torch.float16,
            'fp32': torch.float32,
            'bf16': torch.bfloat16,
            'auto': 'auto',
        }
        dtype = dtype_map.get(kwargs.get('dtype', config.dtype), 'auto')
        
        device = kwargs.get('device', config.device)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use quantization if enabled
        if config.quantize and device == 'cuda':
            try:
                return self.quantizer.load_quantized(model_name)
            except Exception as e:
                log.warning(f"Quantized loading failed: {e}, falling back to standard")
        
        # Standard loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype if dtype != 'auto' else None,
            device_map='auto' if device == 'cuda' else None,
        )
        
        if device == 'cpu':
            model = model.float()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        log.info(f"Transformers model loaded: {model_name}")
        return model, tokenizer
    
    def load_vision_model(self, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """Load a vision-language model."""
        model_name = model_name or config.vision_model
        log.info(f"Loading vision model: {model_name}")
        
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None,
            )
            log.info(f"Vision model loaded: {model_name}")
            return model, processor
        except Exception as e:
            log.error(f"Failed to load vision model: {e}")
            raise
    
    def load_stt_model(self, model_name: Optional[str] = None) -> Any:
        """Load speech-to-text model (Whisper)."""
        model_name = model_name or config.stt_model
        log.info(f"Loading STT model: {model_name}")
        
        try:
            import whisper
            model = whisper.load_model(model_name.split('/')[-1] or "tiny")
            log.info(f"STT model loaded: {model_name}")
            return model
        except ImportError:
            log.warning("openai-whisper not installed, using transformers Whisper")
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            return (model, processor)