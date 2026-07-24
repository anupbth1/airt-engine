"""AIRT-Engine Benchmark - Test accuracy, speed, and memory"""
import time
import torch
from typing import Dict, List, Optional, Any, Callable
from airt.utils.logger import log
from airt.utils.config import config


# Standard test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to check if a number is prime.",
    "What is the meaning of life?",
    "Translate 'Hello, how are you?' to Hindi.",
    "Solve for x: 2x + 5 = 15",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "Explain the theory of relativity.",
    "Write a haiku about artificial intelligence.",
]


class Benchmark:
    """Benchmark model performance with standardized tests."""
    
    def __init__(self):
        self.results = {}
    
    def run_speed_test(self, generate_fn: Callable, 
                       model_name: str = "unknown",
                       num_runs: int = 3) -> Dict[str, Any]:
        """
        Measure inference speed.
        
        Args:
            generate_fn: Function that takes prompt and returns response
            model_name: Name for logging
            num_runs: Number of times to run each prompt
            
        Returns:
            Speed metrics
        """
        log.info(f"Running speed test for {model_name}")
        
        latencies = []
        tokens_counts = []
        
        for prompt in TEST_PROMPTS[:5]:
            for _ in range(num_runs):
                start = time.perf_counter()
                response = generate_fn(prompt)
                elapsed = time.perf_counter() - start
                
                latencies.append(elapsed)
                tokens_counts.append(len(response.split()))
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_counts) / len(tokens_counts)
        tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
        
        results = {
            'model': model_name,
            'avg_latency_s': round(avg_latency, 3),
            'avg_response_tokens': round(avg_tokens, 1),
            'tokens_per_second': round(tokens_per_sec, 2),
            'min_latency_s': round(min(latencies), 3),
            'max_latency_s': round(max(latencies), 3),
        }
        
        log.info(f"Speed test: {results['tokens_per_second']} tok/s avg")
        self.results['speed'] = results
        return results
    
    def run_accuracy_test(self, generate_fn: Callable,
                          model_name: str = "unknown") -> Dict[str, Any]:
        """
        Simple accuracy test using known-answer prompts.
        
        Args:
            generate_fn: Function that takes prompt and returns response
            model_name: Name for logging
            
        Returns:
            Accuracy metrics
        """
        log.info(f"Running accuracy test for {model_name}")
        
        test_cases = [
            ("What is 2 + 2?", ["4", "four"]),
            ("What is the capital of France?", ["Paris"]),
            ("Who wrote Romeo and Juliet?", ["William Shakespeare", "Shakespeare"]),
            ("What is the chemical symbol for water?", ["H2O", "H₂O"]),
            ("How many sides does a triangle have?", ["3", "three"]),
        ]
        
        correct = 0
        total = len(test_cases)
        details = []
        
        for prompt, expected_answers in test_cases:
            response = generate_fn(prompt)
            is_correct = any(answer.lower() in response.lower() for answer in expected_answers)
            
            if is_correct:
                correct += 1
            
            details.append({
                'prompt': prompt,
                'expected': expected_answers,
                'response': response[:100],
                'correct': is_correct,
            })
        
        accuracy = (correct / total) * 100
        
        results = {
            'model': model_name,
            'accuracy_pct': round(accuracy, 1),
            'correct': correct,
            'total': total,
            'details': details,
        }
        
        log.info(f"Accuracy test: {accuracy:.1f}% ({correct}/{total})")
        self.results['accuracy'] = results
        return results
    
    def run_memory_test(self, model) -> Dict[str, Any]:
        """Measure memory usage of a model."""
        log.info("Running memory test")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            results = {
                'device': 'cuda',
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'peak_gb': round(max_allocated, 2),
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            results = {
                'device': 'cpu',
                'rss_gb': round(memory_info.rss / 1e9, 2),
                'vms_gb': round(memory_info.vms / 1e9, 2),
            }
        
        log.info(f"Memory: {results}")
        self.results['memory'] = results
        return results
    
    def full_benchmark(self, generate_fn: Callable, model,
                       model_name: str = "unknown") -> Dict[str, Any]:
        """
        Run complete benchmark: speed, accuracy, memory.
        
        Args:
            generate_fn: Text generation function
            model: Model object (for memory profiling)
            model_name: Model name
            
        Returns:
            Complete benchmark results
        """
        log.info(f"=== Full Benchmark: {model_name} ===")
        
        speed = self.run_speed_test(generate_fn, model_name)
        accuracy = self.run_accuracy_test(generate_fn, model_name)
        memory = self.run_memory_test(model)
        
        # Calculate efficiency score
        efficiency = {
            'accuracy_per_gb': round(accuracy['accuracy_pct'] / max(memory.get('allocated_gb', 1), 0.1), 2),
            'tokens_per_gb_per_sec': round(speed['tokens_per_second'] / max(memory.get('allocated_gb', 1), 0.1), 2),
        }
        
        summary = {
            'model': model_name,
            'accuracy_pct': accuracy['accuracy_pct'],
            'tokens_per_sec': speed['tokens_per_second'],
            'latency_s': speed['avg_latency_s'],
            'memory_gb': memory.get('allocated_gb', memory.get('rss_gb', 0)),
            **efficiency,
        }
        
        self.results['summary'] = summary
        
        log.info(f"=== Results: {accuracy['accuracy_pct']}% accuracy, "
                 f"{speed['tokens_per_second']} tok/s, "
                 f"{summary['memory_gb']} GB ===")
        
        return {
            'summary': summary,
            'speed': speed,
            'accuracy': accuracy,
            'memory': memory,
            'efficiency': efficiency,
        }
    
    def print_report(self, results: Dict[str, Any]):
        """Print benchmark report in readable format."""
        summary = results.get('summary', {})
        speed = results.get('speed', {})
        accuracy = results.get('accuracy', {})
        memory = results.get('memory', {})
        
        print("\n" + "=" * 60)
        print(f" AIRT BENCHMARK REPORT: {summary.get('model', 'Unknown')}")
        print("=" * 60)
        print(f"  Accuracy:    {summary.get('accuracy_pct', 'N/A')}%")
        print(f"  Speed:       {summary.get('tokens_per_sec', 'N/A')} tokens/sec")
        print(f"  Latency:     {summary.get('latency_s', 'N/A')}s avg")
        print(f"  Memory:      {summary.get('memory_gb', 'N/A')} GB")
        print(f"  Efficiency:  {summary.get('accuracy_per_gb', 'N/A')} acc/GB")
        print("-" * 60)
        
        if accuracy.get('details'):
            print("  Accuracy Details:")
            for d in accuracy['details'][:3]:
                status = "✓" if d['correct'] else "✗"
                print(f"    {status} {d['prompt']}")
        
        print("=" * 60 + "\n")