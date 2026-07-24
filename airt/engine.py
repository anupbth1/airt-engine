"""
AIRT-Engine: Main Engine
Adaptive Intelligence Runtime Engine - Unified multi-modal AI system

This is the main entry point that ties together all components:
- Model loading (with auto-backend detection)
- Multi-modal inference (text, vision, audio)
- Optimization (quantization, KV cache, speculative decoding)
- Benchmarking (speed, accuracy, memory)
- Server + CLI interfaces
"""

import os
import sys
from typing import Optional, Dict, Any, Union, Generator
from airt.utils.logger import log, setup_logger
from airt.utils.config import config, AIRTConfig
from airt.inference.runner import InferenceRunner
from airt.inference.benchmark import Benchmark
from airt.models.loader import ModelLoader
from airt.models.profile import ModelProfiler
from airt.optimizers.quantizer import Quantizer
from airt.optimizers.kv_cache import H2OCache, StreamingLLMCache
from airt.optimizers.speculative import SpeculativeDecoder

__version__ = "1.0.0"


class AIRTEngine:
    """
    AIRT-Engine: Main class that orchestrates all components.
    
    Usage:
        engine = AIRTEngine()
        response = engine.chat("What is quantum computing?")
        description = engine.analyze_image("photo.jpg")
        transcript = engine.transcribe("speech.mp3")
        audio = engine.speak("Hello world")
        results = engine.benchmark()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AIRT Engine.
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        if config_path:
            loaded_config = AIRTConfig.from_yaml(config_path)
            for k, v in loaded_config.__dict__.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        
        log.info(f"AIRT-Engine v{__version__} initializing...")
        log.info(f"Device: {config.device}, Backend: {config.backend}")
        log.info(f"Quantization: {config.quantize} ({config.quantization_method}, {config.quantization_bits}-bit)")
        
        self.runner = InferenceRunner()
        self.benchmark = Benchmark()
        self.profiler = ModelProfiler()
        self.quantizer = Quantizer()
        self.loader = ModelLoader()
        
        log.info("AIRT-Engine initialized successfully")
    
    def chat(self, prompt: str, **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Chat with the AI (text only).
        
        Args:
            prompt: Input text
            **kwargs: Model parameters (model_name, max_tokens, temperature, etc.)
            
        Returns:
            Generated response text
        """
        log.info(f"Chat request ({len(prompt)} chars)")
        return self.runner.generate_text(prompt, **kwargs)
    
    def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.",
                      **kwargs) -> str:
        """
        Analyze an image using vision-language model.
        
        Args:
            image_path: Path to image file
            prompt: Question about the image
            **kwargs: Additional parameters
            
        Returns:
            Image analysis
        """
        log.info(f"Image analysis: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.runner.analyze_image(image_path, prompt, **kwargs.get('model_name'))
    
    def transcribe(self, audio_path: str, **kwargs) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            Transcribed text
        """
        log.info(f"Audio transcription: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        return self.runner.transcribe_audio(audio_path, **kwargs.get('model_name'))
    
    def speak(self, text: str, lang: str = "en") -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            lang: Language code (en, hi, zh, ja, ko)
            
        Returns:
            Audio bytes (MP3)
        """
        log.info(f"TTS: {len(text)} chars in {lang}")
        return self.runner.text_to_speech(text, lang)
    
    def benchmark(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full benchmark on a model.
        
        Args:
            model_name: Model to benchmark (default from config)
            
        Returns:
            Benchmark results
        """
        model_name = model_name or config.model_name
        log.info(f"Running full benchmark: {model_name}")
        
        # Load model
        model, _ = self.loader.load_text_model(model_name)
        
        # Create generate function
        def generate_fn(prompt):
            return self.runner.generate_text(prompt, model_name=model_name, 
                                            max_tokens=100, temperature=0.0)
        
        # Run benchmark
        results = self.benchmark.full_benchmark(generate_fn, model, model_name)
        self.benchmark.print_report(results)
        
        return results
    
    def profile(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Profile model requirements and performance.
        
        Args:
            model_name: Model name or size in billions
            
        Returns:
            Profile information
        """
        if model_name and model_name.replace('.', '').isdigit():
            size_b = float(model_name)
            return self.profiler.estimate_gpu_requirements(size_b)
        
        model_name = model_name or config.model_name
        log.info(f"Estimating requirements for: {model_name}")
        
        # Subtract known size from model name
        size_b = 7  # Default guess
        for token in model_name.lower().split('-'):
            if token.endswith('b'):
                try:
                    size_b = float(token[:-1])
                except ValueError:
                    pass
            elif token.endswith('m'):
                try:
                    size_b = float(token[:-1]) / 1000
                except ValueError:
                    pass
        
        return self.profiler.estimate_gpu_requirements(size_b)
    
    def info(self) -> Dict[str, Any]:
        """Get engine information and configuration."""
        import torch
        
        info = {
            'version': __version__,
            'config': config.to_dict(),
            'hardware': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        }
        
        if torch.cuda.is_available():
            info['hardware']['gpu_name'] = torch.cuda.get_device_name(0)
            info['hardware']['vram_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        
        return info
    
    def multi_modal_query(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle multi-modal input and return appropriate response.
        
        Args:
            inputs: Dictionary with keys like 'text', 'image', 'audio'
            
        Returns:
            Dictionary with response
        """
        result = {}
        
        if 'text' in inputs:
            result['text_response'] = self.chat(inputs['text'])
        
        if 'image' in inputs:
            prompt = inputs.get('query', 'Describe this image in detail.')
            result['image_analysis'] = self.analyze_image(inputs['image'], prompt)
        
        if 'audio' in inputs:
            result['transcription'] = self.transcribe(inputs['audio'])
        
        return result


# Factory function
def create_engine(config_path: Optional[str] = None) -> AIRTEngine:
    """Create and return an AIRTEngine instance."""
    return AIRTEngine(config_path)