"""AIRT-Engine Model Profiler - Measures compute, memory, and speed"""
import time
import torch
from typing import Dict, Any, Optional, Tuple
from airt.utils.logger import log
from airt.utils.config import config


class ModelProfiler:
    """Profile model performance: FLOPs, memory, latency."""
    
    def __init__(self):
        self.results = {}
    
    def profile_inference(self, model_fn, input_ids: torch.Tensor, 
                          max_tokens: int = 50, warmup: int = 3) -> Dict[str, Any]:
        """
        Profile model inference.
        
        Args:
            model_fn: Function that generates text (input_ids -> output_ids)
            input_ids: Input tensor
            max_tokens: Number of tokens to generate
            warmup: Number of warmup runs
            
        Returns:
            Dictionary with profile results
        """
        device = input_ids.device
        
        # Warmup
        for _ in range(warmup):
            _ = model_fn(input_ids[:, :10])
        
        # Measure latency
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.perf_counter()
        start_mem = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        output = model_fn(input_ids, max_new_tokens=max_tokens)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        end_mem = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        # Calculate metrics
        total_time = end_time - start_time
        output_tokens = output.shape[1] if hasattr(output, 'shape') else max_tokens
        tokens_per_sec = output_tokens / total_time
        time_per_token = total_time / output_tokens
        memory_used = (end_mem - start_mem) / 1e6 if device.type == 'cuda' else 0
        
        # Estimate FLOPs (rough approximation)
        hidden_dim = 4096  # Typical for 7B models
        num_layers = 32
        flops_per_token = 2 * num_layers * (hidden_dim ** 2) * 4  # Approximate
        total_flops = flops_per_token * output_tokens
        
        results = {
            'total_time_s': round(total_time, 3),
            'tokens_per_sec': round(tokens_per_sec, 2),
            'time_per_token_ms': round(time_per_token * 1000, 2),
            'output_tokens': output_tokens,
            'memory_mb': round(memory_used, 2) if device.type == 'cuda' else 'N/A (CPU)',
            'estimated_gflops': round(total_flops / 1e9, 2),
            'device': device.type,
        }
        
        log.info(f"Profile results: {results['tokens_per_sec']} tok/s, {results['time_per_token_ms']} ms/tok")
        
        return results
    
    def compare_optimizations(self, model_fn, input_ids, optimizations: Dict[str, Any]) -> Dict[str, Dict]:
        """Compare different optimization strategies."""
        results = {}
        
        for name, opt_fn in optimizations.items():
            log.info(f"Profiling: {name}")
            results[name] = self.profile_inference(opt_fn, input_ids)
        
        return results
    
    def estimate_gpu_requirements(self, model_size_b: float) -> Dict[str, float]:
        """Estimate GPU requirements for a model size."""
        # Rough estimates
        fp16_memory = model_size_b * 2  # 2 bytes per param
        fp32_memory = model_size_b * 4
        int4_memory = model_size_b * 0.5
        int8_memory = model_size_b * 1
        
        return {
            'model_size_b': model_size_b,
            'fp16_gb': round(fp16_memory, 2),
            'fp32_gb': round(fp32_memory, 2),
            'int4_gb': round(int4_memory, 2),
            'int8_gb': round(int8_memory, 2),
        }