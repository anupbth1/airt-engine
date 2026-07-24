"""AIRT-Engine KV Cache Optimizer - H2O, StreamingLLM, KVQuant"""
import torch
from typing import Optional, List, Tuple
from airt.utils.logger import log
from airt.utils.config import config


class H2OCache:
    """
    Heavy Hitter Oracle (H2O) KV Cache Eviction.
    Paper: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
    Keeps only the most important tokens in KV cache.
    """
    
    def __init__(self, cache_size: Optional[int] = None):
        self.cache_size = cache_size or config.kv_cache_size
        self.k_cache = None
        self.v_cache = None
        self.token_indices = []
        log.info(f"H2O Cache initialized with size={self.cache_size}")
    
    def evict(self, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Evict less important tokens from KV cache.
        
        Args:
            k: Key tensor [batch, heads, seq_len, dim]
            v: Value tensor [batch, heads, seq_len, dim]
            scores: Attention scores indicating token importance
            
        Returns:
            (k_evicted, v_evicted, important_indices)
        """
        seq_len = k.shape[2]
        
        if seq_len <= self.cache_size:
            return k, v, list(range(seq_len))
        
        # Score: cumulative attention weight for each token
        token_scores = scores.sum(dim=(0, 1, 3))  # [seq_len]
        
        # Keep top-k tokens
        _, important_indices = torch.topk(token_scores, self.cache_size, dim=0)
        important_indices = important_indices.sort().values
        
        k_evicted = k[:, :, important_indices, :]
        v_evicted = v[:, :, important_indices, :]
        
        return k_evicted, v_evicted, important_indices.tolist()


class StreamingLLMCache:
    """
    StreamingLLM KV Cache Management.
    Paper: "Efficient Streaming Language Models with Attention Sinks"
    Keeps initial tokens (attention sinks) + recent tokens + heavy hitters.
    """
    
    def __init__(self, sink_size: int = 4, recent_size: int = 256, cache_size: Optional[int] = None):
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.max_cache = cache_size or config.kv_cache_size
        self.heavy_hitter_size = self.max_cache - sink_size - recent_size
        
        self.k_cache = None
        self.v_cache = None
        self.sink_indices = []
        self.recent_indices = []
        self.heavy_indices = []
        
        log.info(f"StreamingLLM Cache: sink={sink_size}, recent={recent_size}, heavy={self.heavy_hitter_size}")
    
    def update(self, k: torch.Tensor, v: torch.Tensor, 
               scores: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with StreamingLLM strategy.
        
        Args:
            k: Key tensor
            v: Value tensor
            scores: Attention scores
            step: Current generation step
            
        Returns:
            (k_updated, v_updated)
        """
        seq_len = k.shape[2]
        
        if seq_len <= self.max_cache:
            return k, v
        
        # Keep first sink_size tokens (attention sinks)
        sink_k = k[:, :, :self.sink_size, :]
        sink_v = v[:, :, :self.sink_size, :]
        
        # Keep recent tokens
        recent_k = k[:, :, -self.recent_size:, :]
        recent_v = v[:, :, -self.recent_size:, :]
        
        # For middle tokens, keep heavy hitters
        middle_start = self.sink_size
        middle_end = seq_len - self.recent_size
        
        if middle_end > middle_start and self.heavy_hitter_size > 0:
            middle_k = k[:, :, middle_start:middle_end, :]
            middle_v = v[:, :, middle_start:middle_end, :]
            
            # Score middle tokens
            middle_scores = scores[:, :, middle_start:middle_end].sum(dim=(0, 1, 3))
            
            if middle_k.shape[2] > self.heavy_hitter_size:
                _, heavy_idx = torch.topk(middle_scores, self.heavy_hitter_size, dim=0)
                heavy_idx = heavy_idx.sort().values
                
                heavy_k = middle_k[:, :, heavy_idx, :]
                heavy_v = middle_v[:, :, heavy_idx, :]
            else:
                heavy_k = middle_k
                heavy_v = middle_v
        else:
            heavy_k = torch.empty_like(k[:, :, :0, :])
            heavy_v = torch.empty_like(v[:, :, :0, :])
        
        # Combine
        k_updated = torch.cat([sink_k, heavy_k, recent_k], dim=2)
        v_updated = torch.cat([sink_v, heavy_v, recent_v], dim=2)
        
        return k_updated, v_updated


def create_kv_cache(method: Optional[str] = None, **kwargs):
    """Factory function for KV cache optimizers."""
    method = method or config.kv_cache_method
    
    if method == "h2o":
        return H2OCache(**kwargs)
    elif method == "streamingllm":
        return StreamingLLMCache(**kwargs)
    else:
        log.warning(f"Unknown KV cache method: {method}, using H2O")
        return H2OCache(**kwargs)