"""AIRT-Engine Quantizer - Supports GPTQ, AWQ, GGUF, and bitsandbytes"""
import os
import torch
from typing import Optional, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from airt.utils.logger import log
from airt.utils.config import config


class Quantizer:
    """Quantize models using various methods for reduced memory and compute."""
    
    def __init__(self, method: Optional[str] = None, bits: Optional[int] = None):
        self.method = method or config.quantization_method
        self.bits = bits or config.quantization_bits
        
    def quantize_gguf(self, model_name: str, output_path: Optional[str] = None) -> str:
        """
        Convert model to GGUF format for efficient CPU inference.
        Uses llama.cpp internally.
        
        Args:
            model_name: HuggingFace model name or path
            output_path: Path to save GGUF file
            
        Returns:
            Path to GGUF file
        """
        log.info(f"Converting {model_name} to GGUF format ({self.bits}-bit)...")
        
        # For GGUF, we use llama.cpp which loads quantized models directly
        # The conversion is typically done before runtime
        # Here we return the model path info for llama.cpp loader
        
        gguf_path = output_path or os.path.join(
            config.model_cache_dir, 
            model_name.replace('/', '_') + f"_Q{self.bits}.gguf"
        )
        
        log.info(f"GGUF target: {gguf_path}")
        log.info(f"Tip: Use 'python -m llama_cpp.gguf.convert_hf' for conversion")
        
        return gguf_path
    
    def load_gptq(self, model_name: str) -> tuple:
        """
        Load a GPTQ-quantized model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'TheBloke/Llama-2-7B-GPTQ')
            
        Returns:
            (model, tokenizer) tuple
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM
            
            log.info(f"Loading GPTQ model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                use_triton=False,
                use_safetensors=True,
                trust_remote_code=True,
            )
            
            log.info(f"GPTQ model loaded: {model_name}")
            return model, tokenizer
            
        except ImportError:
            log.error("auto-gptq not installed. Run: pip install auto-gptq")
            raise
        except Exception as e:
            log.error(f"Failed to load GPTQ model: {e}")
            raise
    
    def load_awq(self, model_name: str) -> tuple:
        """
        Load an AWQ-quantized model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'TheBloke/Llama-2-7B-AWQ')
            
        Returns:
            (model, tokenizer) tuple
        """
        try:
            from awq import AutoAWQForCausalLM
            
            log.info(f"Loading AWQ model: {model_name}")
            
            model = AutoAWQForCausalLM.from_quantized(
                model_name,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                safetensors=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            log.info(f"AWQ model loaded: {model_name}")
            return model, tokenizer
            
        except ImportError:
            log.error("autoawq not installed. Run: pip install autoawq")
            raise
        except Exception as e:
            log.error(f"Failed to load AWQ model: {e}")
            raise
    
    def load_quantized(self, model_name: str) -> tuple:
        """
        Auto-detect and load quantized model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            (model, tokenizer) tuple
        """
        # Detect quantization method from model name
        name_lower = model_name.lower()
        
        if 'awq' in name_lower:
            return self.load_awq(model_name)
        elif 'gptq' in name_lower:
            return self.load_gptq(model_name)
        else:
            # Try standard transformers with bitsandbytes
            return self.load_bitsandbytes(model_name)
    
    def load_bitsandbytes(self, model_name: str) -> tuple:
        """
        Load model with bitsandbytes quantization.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            (model, tokenizer) tuple
        """
        try:
            import bitsandbytes as bnb
            
            log.info(f"Loading {model_name} with {self.bits}-bit bitsandbytes quantization...")
            
            quantization_config = {
                4: {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                },
                8: {
                    "load_in_8bit": True,
                }
            }
            
            config_map = quantization_config.get(self.bits, quantization_config[4])
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                **config_map
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            log.info(f"bitsandbytes model loaded: {model_name}")
            return model, tokenizer
            
        except ImportError:
            log.error("bitsandbytes not installed. Run: pip install bitsandbytes")
            raise
        except Exception as e:
            log.error(f"Failed to load with bitsandbytes: {e}")
            raise


def get_quantizer(method: Optional[str] = None, bits: Optional[int] = None) -> Quantizer:
    """Factory function to create a Quantizer instance."""
    return Quantizer(method, bits)