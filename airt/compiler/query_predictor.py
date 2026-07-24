"""
AIRT-Compiler: Query Cost Predictor
Novel: Predicts compute cost of a query BEFORE running it.
Allows the system to allocate minimum compute for each query.

This is like an "AI fuel gauge" - estimates how much compute
a query will need before actually running it.
"""
import torch
import re
from typing import Dict, List, Tuple, Optional, Any
from airt.utils.logger import log


class QueryCostPredictor:
    """
    Predicts the compute cost of a query without running the model.
    
    Uses:
    1. Query length and complexity
    2. Reasoning keywords presence
    3. Domain detection (math vs casual chat)
    4. Past query patterns (caching)
    """
    
    def __init__(self):
        self.query_history = []
        self.cost_cache = {}
        
        # Keywords that indicate high compute cost
        self.high_cost_keywords = [
            'explain', 'why', 'how', 'calculate', 'solve', 'prove',
            'analyze', 'compare', 'contrast', 'difference between',
            'write code', 'implement', 'debug', 'optimize',
            'quantum', 'relativity', 'philosophy', 'mathematics',
            'derivation', 'proof', 'theorem', 'equation',
        ]
        
        # Keywords that indicate low compute cost
        self.low_cost_keywords = [
            'hello', 'hi', 'yes', 'no', 'thanks', 'ok',
            'who is', 'what is', 'where is', 'when did',
            'capital of', 'population of',
        ]
    
    def predict_cost(self, query: str) -> Dict[str, Any]:
        """
        Predict compute cost for a query.
        
        Args:
            query: User query string
            
        Returns:
            Cost prediction dictionary
        """
        # Check cache
        query_key = query.lower().strip()
        if query_key in self.cost_cache:
            return self.cost_cache[query_key]
        
        # Analysis signals
        query_len = len(query)
        word_count = len(query.split())
        
        # Signal 1: Length-based cost
        length_cost = min(query_len / 1000, 1.0)  # 0 (short) to 1 (long)
        
        # Signal 2: Keyword-based cost
        query_lower = query.lower()
        high_matches = sum(1 for kw in self.high_cost_keywords if kw in query_lower)
        low_matches = sum(1 for kw in self.low_cost_keywords if kw in query_lower)
        
        keyword_cost = min(high_matches / 5, 1.0)  # 0 to 1
        keyword_discount = min(low_matches / 3, 1.0)  # Reduce cost for simple queries
        
        # Signal 3: Question complexity
        has_math = bool(re.search(r'[\+\-\*/\^=√∫∑]', query))
        has_code = bool(re.search(r'(def |class |import |function|lambda)', query))
        has_number = bool(re.search(r'\d+', query))
        
        complexity_cost = 0.0
        if has_math:
            complexity_cost += 0.3
        if has_code:
            complexity_cost += 0.4
        if has_number and has_math:
            complexity_cost += 0.3
        
        # Signal 4: Reasoning depth estimation
        reasoning_depth = 0
        if 'why' in query_lower or 'explain' in query_lower:
            reasoning_depth += 2
        if 'compare' in query_lower or 'contrast' in query_lower:
            reasoning_depth += 2
        if 'prove' in query_lower or 'derive' in query_lower:
            reasoning_depth += 3
        if 'write' in query_lower and 'code' in query_lower:
            reasoning_depth += 2
        
        # Combined cost (0 to 1)
        base_cost = 0.1  # Minimum cost for any query
        variable_cost = (length_cost * 0.2 + 
                        keyword_cost * 0.3 + 
                        complexity_cost * 0.3 + 
                        min(reasoning_depth / 5, 1.0) * 0.2)
        
        # Apply discount for simple queries
        variable_cost = max(variable_cost - keyword_discount * 0.2, 0)
        
        total_cost = min(base_cost + variable_cost, 1.0)
        
        # Determine category
        if total_cost < 0.2:
            category = 'tiny'
            recommended_precision = 'int2'
        elif total_cost < 0.4:
            category = 'simple'
            recommended_precision = 'int4'
        elif total_cost < 0.6:
            category = 'medium'
            recommended_precision = 'int4'
        elif total_cost < 0.8:
            category = 'complex'
            recommended_precision = 'int8'
        else:
            category = 'expert'
            recommended_precision = 'fp16'
        
        prediction = {
            'query': query,
            'query_length': query_len,
            'word_count': word_count,
            'cost_score': round(total_cost, 3),
            'category': category,
            'recommended_precision': recommended_precision,
            'reasoning_depth': reasoning_depth,
            'signals': {
                'length_cost': round(length_cost, 3),
                'keyword_cost': round(keyword_cost, 3),
                'complexity_cost': round(complexity_cost, 3),
                'reasoning_depth': reasoning_depth,
            },
            'estimated_tokens': self._estimate_tokens(total_cost, query_len),
        }
        
        # Cache result
        self.cache_cost(query_key, prediction)
        
        return prediction
    
    def _estimate_tokens(self, cost: float, query_len: int) -> int:
        """Estimate output tokens needed."""
        base_tokens = 20
        cost_tokens = int(cost * 200)  # 0 to 200 extra tokens
        len_tokens = int(query_len / 10)  # 1 token per 10 chars
        return min(base_tokens + cost_tokens + len_tokens, 1024)
    
    def cache_cost(self, query_key: str, prediction: Dict[str, Any]):
        """Cache a cost prediction for reuse."""
        self.cost_cache[query_key] = prediction
        if len(self.cost_cache) > 1000:
            # Evict oldest
            self.cost_cache.pop(next(iter(self.cost_cache)))
    
    def batch_predict(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Predict costs for multiple queries."""
        return [self.predict_cost(q) for q in queries]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        categories = {}
        for pred in self.cost_cache.values():
            cat = pred['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'cached_queries': len(self.cost_cache),
            'category_distribution': categories,
            'avg_cost': sum(p['cost_score'] for p in self.cost_cache.values()) / max(len(self.cost_cache), 1),
        }


# Global predictor
_predictor = None

def get_predictor() -> QueryCostPredictor:
    global _predictor
    if _predictor is None:
        _predictor = QueryCostPredictor()
    return _predictor


def predict_query_cost(query: str) -> Dict[str, Any]:
    """Convenience function to predict query cost."""
    return get_predictor().predict_cost(query)