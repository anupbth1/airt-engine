"""AIRT-Engine Inference Runner - Unified inference for all backends"""
import torch
from typing import Optional, List, Dict, Any, Union, Generator
from transformers import AutoTokenizer, TextStreamer
from airt.utils.logger import log
from airt.utils.config import config
from airt.models.loader import ModelLoader


class InferenceRunner:
    """Unified inference runner for text, vision, and audio models."""
    
    def __init__(self):
        self.loader = ModelLoader()
        self.models = {}
        self.tokenizers = {}
    
    def generate_text(self, 
                      prompt: str,
                      model_name: Optional[str] = None,
                      max_tokens: int = 256,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      stream: bool = False,
                      **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            model_name: Model to use (default from config)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated text or generator if streaming
        """
        model_name = model_name or config.model_name
        model_key = f"text:{model_name}"
        
        if model_key not in self.models:
            model, tokenizer = self.loader.load_text_model(model_name, **kwargs)
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
        else:
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
        
        backend = kwargs.get('backend', config.backend)
        if backend == 'auto':
            backend = self.loader.detect_best_backend()
        
        if backend == 'vllm':
            return self._generate_vllm(model, prompt, max_tokens, temperature, top_p, stream)
        elif backend == 'llama_cpp':
            return self._generate_llamacpp(model, prompt, max_tokens, temperature, top_p, stream)
        else:
            return self._generate_transformers(model, tokenizer, prompt, max_tokens, temperature, top_p, stream)
    
    def _generate_transformers(self, model, tokenizer, prompt, max_tokens, temperature, top_p, stream):
        """Generate using HuggingFace transformers."""
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if stream:
            return self._stream_transformers(model, tokenizer, inputs, max_tokens, temperature, top_p)
        
        with torch.no_grad():
            # transformers v5.x compatibility: use GenerationMixin explicitly
            from transformers import GenerationMixin
            if isinstance(model, GenerationMixin):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elif hasattr(model, 'generate'):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                # Fallback: use pipeline WITHOUT device arg (conflicts with accelerate)
                from transformers import pipeline
                pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
                outputs = pipe(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    return_full_text=False,
                )
                response = outputs[0]['generated_text'].strip()
                return response
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _stream_transformers(self, model, tokenizer, inputs, max_tokens, temperature, top_p):
        """Stream tokens from transformers model."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
    
    def _generate_vllm(self, model, prompt, max_tokens, temperature, top_p, stream):
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def _generate_llamacpp(self, model, prompt, max_tokens, temperature, top_p, stream):
        """Generate using llama.cpp."""
        if stream:
            for output in model(prompt, max_tokens=max_tokens, temperature=temperature, 
                                top_p=top_p, stream=True):
                yield output['choices'][0]['text']
            return
        
        output = model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return output['choices'][0]['text'].strip()
    
    def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.",
                      model_name: Optional[str] = None) -> str:
        """
        Analyze an image using vision-language model.
        
        Args:
            image_path: Path to image file
            prompt: Question about the image
            model_name: Vision model name
            
        Returns:
            Model's description/analysis
        """
        model_name = model_name or config.vision_model
        model_key = f"vision:{model_name}"
        
        if model_key not in self.models:
            model, processor = self.loader.load_vision_model(model_name)
            self.models[model_key] = model
            self.tokenizers[model_key] = processor
        else:
            model = self.models[model_key]
            processor = self.tokenizers[model_key]
        
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Handle different vision model types
        if hasattr(processor, 'apply_chat_template'):
            # Modern VLM (e.g., LLaVA, MiniCPM)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=inputs, images=image, return_tensors="pt")
        else:
            # Simpler VLM (e.g., moondream2)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def transcribe_audio(self, audio_path: str, model_name: Optional[str] = None) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            model_name: Whisper model name
            
        Returns:
            Transcribed text
        """
        model_name = model_name or config.stt_model
        model_key = f"stt:{model_name}"
        
        if model_key not in self.models:
            model = self.loader.load_stt_model(model_name)
            self.models[model_key] = model
        
        model = self.models[model_key]
        
        if isinstance(model, tuple):
            # HuggingFace Whisper
            whisper_model, processor = model
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = whisper_model.generate(inputs["input_features"])
            
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # OpenAI Whisper
            result = model.transcribe(audio_path)
            return result['text'].strip()
    
    def text_to_speech(self, text: str, lang: str = "en") -> bytes:
        """
        Convert text to speech using edge-tts (free, no GPU needed).
        
        Args:
            text: Text to speak
            lang: Language code
            
        Returns:
            Audio bytes (MP3)
        """
        import asyncio
        import edge_tts
        
        voice_map = {
            "en": "en-US-ChristopherNeural",
            "hi": "hi-IN-SwaraNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
        }
        
        voice = voice_map.get(lang, "en-US-ChristopherNeural")
        
        async def _tts():
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        return asyncio.run(_tts())