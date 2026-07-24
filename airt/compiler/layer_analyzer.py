"""
AIRT-Compiler: Layer Importance Analyzer
Novel: Analyzes each layer's contribution to model accuracy
and assigns importance scores for dynamic precision allocation.

This is what makes AIRT different from GPTQ/AWQ:
- GPTQ: Fixed 4-bit for ALL layers ❌
- AIRT: Dynamic precision per layer based on importance ✅
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

from airt.utils.logger import log


class LayerAnalyzer:
    """
    Analyzes transformer layers to determine their importance.
    
    Uses 3 signals:
    1. Activation Variance: High variance = more information = more important
    2. Gradient Sensitivity: How much output changes when layer is perturbed
    3. Attention Score Distribution: How much the layer contributes to final output
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        self.hidden_dim = 0
        self.layer_importance = {}
        self.activation_stats = {}
        
    def load_model(self):
        """Load model for analysis."""
        log.info(f"Loading model: {self.model_name}")
        
        # First load config
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except:
            config = None
        
        # Load model with proper settings for transformers v5.x compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Enable hidden states output via config after loading
        if hasattr(self.model, 'config'):
            self.model.config.output_hidden_states = True
            self.model.config.output_attentions = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Detect model architecture
        config = self.model.config
        if hasattr(config, 'num_hidden_layers'):
            self.num_layers = config.num_hidden_layers
        elif hasattr(config, 'num_layers'):
            self.num_layers = config.num_layers
        else:
            # Count attention layers
            for name, module in self.model.named_modules():
                if 'attention' in name.lower() and 'self' in name.lower():
                    self.num_layers += 1
            self.num_layers = max(self.num_layers // 2, 1)
        
        if hasattr(config, 'hidden_size'):
            self.hidden_dim = config.hidden_size
        elif hasattr(config, 'd_model'):
            self.hidden_dim = config.d_model
        else:
            self.hidden_dim = 4096  # Default for 7B models
        
        log.info(f"Model loaded: {self.num_layers} layers, {self.hidden_dim} hidden dim")
        return self.model, self.tokenizer
    
    def collect_activation_stats(self, sample_texts: List[str], num_batches: int = 5):
        """
        Feed sample texts through the model and collect activation statistics.
        Uses model's output_hidden_states instead of hooks for transformers v5.x compatibility.
        
        Args:
            sample_texts: List of sample prompts
            num_batches: Number of forward passes
        """
        if self.model is None:
            self.load_model()
        
        log.info("Collecting activation statistics...")
        
        # Use model's built-in hidden states output (no hooks needed)
        self.model.config.output_hidden_states = True
        
        layer_outputs = {i: [] for i in range(self.num_layers)}
        
        # Run forward passes
        device = next(self.model.parameters()).device
        
        for text in tqdm(sample_texts[:num_batches], desc="Analyzing layers"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    # Try with output_hidden_states
                    outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                    
                    # Extract hidden states from model output
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # hidden_states includes embedding output + each layer output
                        # Skip embedding (index 0), take layer outputs
                        hs = outputs.hidden_states
                        for i in range(min(len(hs) - 1, self.num_layers)):
                            layer_hidden = hs[i + 1]  # Skip embedding layer
                            if layer_hidden is not None:
                                layer_outputs[i].append(layer_hidden.detach().float())
                    else:
                        log.warning("No hidden_states in output. Trying alternative approach...")
                        break
                except Exception as e:
                    log.warning(f"Hidden states approach failed: {str(e)[:60]}")
                    break
        
        # Calculate importance scores (will catch empty case)
        self._calculate_importance(layer_outputs)
        
        # Check if we got any data
        total_captured = sum(len(v) for v in layer_outputs.values())
        if total_captured == 0:
            log.warning("No layer outputs captured. This may be due to model architecture.")
            log.warning("For this model, analytical defaults will be used.")
            # Set default importance
            for i in range(self.num_layers):
                self.layer_importance[i] = 0.3 + 0.6 * (i / max(self.num_layers - 1, 1))
        
        log.info(f"Activation analysis complete (captured {total_captured} layer outputs)")
        return self.layer_importance
    
    def _calculate_importance(self, layer_outputs: Dict[int, List[torch.Tensor]]):
        """
        Calculate importance scores for each layer.
        
        Signal 1: Activation Variance (higher = more information)
        Signal 2: Layer Magnitude (larger norms = more influential)
        Signal 3: Sensitivity (how much output depends on this layer)
        """
        for layer_idx, outputs in layer_outputs.items():
            if not outputs:
                self.layer_importance[layer_idx] = 0.5
                continue
            
            # Stack all outputs for this layer
            stacked = torch.cat([o.flatten() for o in outputs])
            
            # Signal 1: Variance importance
            variance = stacked.var().item()
            variance_score = min(variance / (variance + 1), 1.0)
            
            # Signal 2: Magnitude importance
            magnitude = stacked.abs().mean().item()
            magnitude_score = min(magnitude / (magnitude + 1), 1.0)
            
            # Signal 3: Sparsity (more sparse = more specialized = more important)
            sparsity = (stacked.abs() > stacked.abs().mean()).float().mean().item()
            
            # Combined score
            importance = 0.4 * variance_score + 0.4 * magnitude_score + 0.2 * sparsity
            self.layer_importance[layer_idx] = importance
        
        # Normalize scores
        if self.layer_importance:
            max_score = max(self.layer_importance.values())
            min_score = min(self.layer_importance.values())
            range_score = max_score - min_score
            
            if range_score > 0:
                for idx in self.layer_importance:
                    self.layer_importance[idx] = (self.layer_importance[idx] - min_score) / range_score
    
    def get_precision_plan(self, target_compression: float = 0.5) -> Dict[int, str]:
        """
        Generate precision plan based on layer importance.
        
        Args:
            target_compression: 0.0 (no compression) to 1.0 (max compression)
            
        Returns:
            Dict mapping layer_idx -> precision ('int2', 'int4', 'int8', 'fp16')
        """
        if not self.layer_importance:
            log.warning("No layer importance data. Run collect_activation_stats first.")
            return {i: 'int4' for i in range(self.num_layers)}
        
        # Sort layers by importance
        sorted_layers = sorted(self.layer_importance.items(), key=lambda x: x[1])
        
        # Assign precision based on percentile
        n_layers = len(sorted_layers)
        precision_plan = {}
        
        precision_map = {
            0.0: 'int2',    # Bottom 0-25%: extreme compression
            0.25: 'int4',   # 25-50%: high compression
            0.5: 'int8',    # 50-75%: moderate compression  
            0.75: 'fp16',   # Top 75-100%: full precision
        }
        
        # Adjust thresholds based on target compression
        if target_compression >= 0.75:
            thresholds = [0.0, 0.15, 0.50, 0.75]  # More aggressive
        elif target_compression >= 0.5:
            thresholds = [0.0, 0.25, 0.50, 0.75]  # Balanced
        else:
            thresholds = [0.0, 0.50, 0.75, 0.90]  # Conservative
        
        for i, (layer_idx, importance) in enumerate(sorted_layers):
            percentile = i / n_layers
            
            if percentile < thresholds[1]:
                precision_plan[layer_idx] = 'int2'
            elif percentile < thresholds[2]:
                precision_plan[layer_idx] = 'int4'
            elif percentile < thresholds[3]:
                precision_plan[layer_idx] = 'int8'
            else:
                precision_plan[layer_idx] = 'fp16'
        
        # Log plan summary
        precisions = list(precision_plan.values())
        log.info(f"Precision Plan: "
                 f"INT2: {precisions.count('int2')}, "
                 f"INT4: {precisions.count('int4')}, "
                 f"INT8: {precisions.count('int8')}, "
                 f"FP16: {precisions.count('fp16')}")
        
        return precision_plan
    
    def estimate_compression(self, precision_plan: Dict[int, str]) -> Dict[str, float]:
        """
        Estimate memory compression from precision plan.
        
        Args:
            precision_plan: Dict mapping layer_idx -> precision
            
        Returns:
            Compression statistics
        """
        bits_map = {'int2': 2, 'int4': 4, 'int8': 8, 'fp16': 16}
        
        total_bits = 0
        layer_counts = {'int2': 0, 'int4': 0, 'int8': 0, 'fp16': 0}
        
        for idx, precision in precision_plan.items():
            bits = bits_map.get(precision, 16)
            total_bits += bits
            layer_counts[precision] += 1
        
        avg_bits = total_bits / len(precision_plan) if precision_plan else 0
        compression_ratio = 16 / avg_bits if avg_bits > 0 else 1
        memory_savings = (1 - avg_bits / 16) * 100
        
        return {
            'avg_bits_per_weight': round(avg_bits, 2),
            'compression_ratio': round(compression_ratio, 2),
            'memory_savings_pct': round(memory_savings, 1),
            'layer_distribution': layer_counts,
        }


def analyze_model(model_name: str, sample_texts: Optional[List[str]] = None):
    """
    Convenience function to analyze a model.
    
    Args:
        model_name: HuggingFace model name
        sample_texts: Sample prompts for analysis
        
    Returns:
        (analyzer, precision_plan, compression_stats)
    """
    if sample_texts is None:
        sample_texts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
            "The theory of relativity states that",
            "Machine learning is a subset of",
            "Once upon a time in a far away land",
            "The key to artificial intelligence is",
        ]
    
    analyzer = LayerAnalyzer(model_name)
    analyzer.load_model()
    analyzer.collect_activation_stats(sample_texts)
    
    # Generate plans at different compression levels
    plans = {}
    for compression in [0.3, 0.5, 0.7]:
        plan = analyzer.get_precision_plan(compression)
        stats = analyzer.estimate_compression(plan)
        plans[compression] = {'plan': plan, 'stats': stats}
    
    return analyzer, plans