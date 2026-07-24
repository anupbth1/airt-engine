"""AIRT-Engine Speculative Decoding - Draft-Verify Strategy"""
from typing import Optional, List, Tuple, Callable
import torch
from airt.utils.logger import log
from airt.utils.config import config


class SpeculativeDecoder:
    """
    Speculative Decoding: Small draft model generates candidates,
    big model verifies them in parallel.
    
    Paper: "Fast Inference from Transformers via Speculative Decoding"
    """
    
    def __init__(self, 
                 draft_model_fn: Callable,
                 target_model_fn: Callable,
                 draft_rate: int = 3,
                 temperature: float = 0.0):
        """
        Args:
            draft_model_fn: Function that runs small model (input_ids -> logits)
            target_model_fn: Function that runs big model (input_ids -> logits)
            draft_rate: Number of tokens to draft per verification step
            temperature: Sampling temperature (0 = greedy)
        """
        self.draft_model_fn = draft_model_fn
        self.target_model_fn = target_model_fn
        self.draft_rate = draft_rate
        self.temperature = temperature
        log.info(f"Speculative Decoder initialized with draft_rate={draft_rate}")
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated token IDs
        """
        all_ids = input_ids.clone()
        device = input_ids.device
        generated = 0
        
        while generated < max_new_tokens:
            current_len = all_ids.shape[1]
            remaining = max_new_tokens - generated
            current_draft = min(self.draft_rate, remaining)
            
            # Step 1: Draft tokens with small model
            draft_ids = all_ids.clone()
            for _ in range(current_draft):
                draft_logits = self.draft_model_fn(draft_ids)
                draft_token = self._sample(draft_logits[:, -1, :])
                draft_ids = torch.cat([draft_ids, draft_token], dim=1)
            
            # Step 2: Verify with target model (single forward pass)
            target_logits = self.target_model_fn(draft_ids)
            
            # Step 3: Accept/reject tokens
            accept_count = 0
            for i in range(current_draft):
                draft_token_id = draft_ids[:, current_len + i]
                target_prob = torch.softmax(target_logits[:, current_len + i - 1, :], dim=-1)
                draft_prob = torch.softmax(draft_logits[:, -1, :], dim=-1) if i == 0 else \
                    torch.softmax(self.draft_model_fn(draft_ids[:, :current_len + i])[:, -1, :], dim=-1)
                
                # Rejection sampling
                p_draft = draft_prob.gather(-1, draft_token_id.unsqueeze(-1)).squeeze(-1)
                p_target = target_prob.gather(-1, draft_token_id.unsqueeze(-1)).squeeze(-1)
                
                if torch.rand(1, device=device).item() < (p_target / p_draft).item():
                    accept_count += 1
                else:
                    # Reject and resample from adjusted distribution
                    adjusted_probs = torch.relu(target_prob - draft_prob)
                    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
                    new_token = self._sample(adjusted_probs)
                    draft_ids = draft_ids[:, :current_len + i]
                    draft_ids = torch.cat([draft_ids, new_token], dim=1)
                    break
            
            # Take accepted tokens
            all_ids = draft_ids[:, :current_len + accept_count + 1]
            generated += accept_count + 1
        
        return all_ids[:, input_ids.shape[1]:]
    
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self.temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)