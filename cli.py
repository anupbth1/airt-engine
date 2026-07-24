"""
AIRT-Engine CLI - Command-line interface for multi-modal AI

Usage:
  python cli.py chat "What is quantum computing?"
  python cli.py image photo.jpg "Describe this"
  python cli.py transcribe speech.mp3
  python cli.py speak "Hello world" --lang hi
  python cli.py benchmark --model microsoft/Phi-3.5-mini-instruct
  python cli.py compiler predict "Hello world"
  python cli.py compiler demo
  python cli.py server
"""
import argparse
import sys
import os
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="AIRT-Engine: Adaptive Intelligence Runtime Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py chat "Explain quantum computing"
  python cli.py image photo.jpg "What is this?"
  python cli.py transcribe recording.mp3
  python cli.py speak "Namaste" --lang hi
  python cli.py benchmark --model microsoft/Phi-3.5-mini-instruct
  python cli.py compiler predict "Hello"
  python cli.py compiler demo
  python cli.py server --port 8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with the AI')
    chat_parser.add_argument('prompt', type=str, help='Your message')
    chat_parser.add_argument('--model', '-m', type=str, default=None, help='Model name')
    chat_parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens')
    chat_parser.add_argument('--temperature', '-t', type=float, default=0.7, help='Temperature')
    
    # Image command
    image_parser = subparsers.add_parser('image', help='Analyze an image')
    image_parser.add_argument('image_path', type=str, help='Path to image file')
    image_parser.add_argument('prompt', type=str, nargs='?', default="Describe this image in detail.",
                             help='Question about the image')
    image_parser.add_argument('--model', type=str, default=None, help='Vision model')
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe audio to text')
    transcribe_parser.add_argument('audio_path', type=str, help='Path to audio file')
    transcribe_parser.add_argument('--model', type=str, default=None, help='Whisper model')
    
    # Speak command
    speak_parser = subparsers.add_parser('speak', help='Text to speech')
    speak_parser.add_argument('text', type=str, help='Text to speak')
    speak_parser.add_argument('--lang', '-l', type=str, default='en', 
                             choices=['en', 'hi', 'zh', 'ja', 'ko'],
                             help='Language')
    speak_parser.add_argument('--output', '-o', type=str, default=None, help='Output file path')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark a model')
    bench_parser.add_argument('--model', '-m', type=str, default=None, help='Model to benchmark')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile model requirements')
    profile_parser.add_argument('--model', '-m', type=str, default=None, 
                               help='Model name or size in billions (e.g., 7)')
    
    # Compiler commands
    compiler_parser = subparsers.add_parser('compiler', help='AIRT-Compiler: Optimize models')
    compiler_sub = compiler_parser.add_subparsers(dest='compiler_cmd')
    compiler_sub.add_parser('demo', help='Show full capabilities demo')
    cp = compiler_sub.add_parser('predict', help='Predict query compute cost')
    cp.add_argument('query', type=str, help='Query to analyze')
    ca = compiler_sub.add_parser('analyze', help='Analyze model layers (needs model)')
    ca.add_argument('model_name', type=str, help='HuggingFace model name')
    cc = compiler_sub.add_parser('compare', help='Compare AIRT vs fixed quantization')
    cc.add_argument('model_name', type=str, help='Model name')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show engine information')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start the web server')
    server_parser.add_argument('--port', '-p', type=int, default=8000, help='Port number')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'server':
        start_server(args)
    elif args.command == 'chat':
        cmd_chat(args)
    elif args.command == 'image':
        cmd_image(args)
    elif args.command == 'transcribe':
        cmd_transcribe(args)
    elif args.command == 'speak':
        cmd_speak(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'profile':
        cmd_profile(args)
    elif args.command == 'compiler':
        cmd_compiler(args)
    elif args.command == 'info':
        cmd_info()


def cmd_compiler(args):
    """Handle AIRT-Compiler commands."""
    if not hasattr(args, 'compiler_cmd') or not args.compiler_cmd:
        print("Compiler commands: demo, predict, analyze, compare")
        print("  python cli.py compiler demo")
        print("  python cli.py compiler predict 'Hello world'")
        print("  python cli.py compiler analyze microsoft/Phi-3.5-mini-instruct")
        print("  python cli.py compiler compare microsoft/Phi-3.5-mini-instruct")
        return
    
    if args.compiler_cmd == 'demo':
        from airt.compiler.compiler_cli import cmd_demo
        cmd_demo()
    elif args.compiler_cmd == 'predict':
        from airt.compiler.compiler_cli import cmd_predict
        cmd_predict(args.query)
    elif args.compiler_cmd == 'analyze':
        from airt.compiler.compiler_cli import cmd_analyze
        cmd_analyze(args.model_name)
    elif args.compiler_cmd == 'compare':
        from airt.compiler.compiler_cli import cmd_compare
        cmd_compare(args.model_name)


def cmd_chat(args):
    """Handle chat command."""
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    print(f"\nPrompt: {args.prompt}")
    print(f"   Model: {args.model or 'default'}")
    print("-" * 50)
    
    response = engine.chat(
        args.prompt,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    print(f"\n{response}\n")


def cmd_image(args):
    """Handle image analysis command."""
    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        return
    
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    print(f"\nAnalyzing: {args.image_path}")
    print(f"   Prompt: {args.prompt}")
    print("-" * 50)
    
    analysis = engine.analyze_image(args.image_path, args.prompt, model_name=args.model)
    print(f"\n{analysis}\n")


def cmd_transcribe(args):
    """Handle audio transcription command."""
    if not os.path.exists(args.audio_path):
        print(f"Audio not found: {args.audio_path}")
        return
    
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    print(f"\nTranscribing: {args.audio_path}")
    print("-" * 50)
    
    transcription = engine.transcribe(args.audio_path, model_name=args.model)
    print(f"\n{transcription}\n")


def cmd_speak(args):
    """Handle text-to-speech command."""
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    print(f"\nConverting to speech: '{args.text[:50]}...' ({args.lang})")
    print("-" * 50)
    
    audio_bytes = engine.speak(args.text, args.lang)
    
    output_path = args.output or f"speech_{args.lang}.mp3"
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)
    
    print(f"Audio saved to: {output_path}\n")


def cmd_benchmark(args):
    """Handle benchmark command."""
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    model_name = args.model
    print(f"\nBenchmarking: {model_name or 'default model'}")
    print("=" * 50)
    
    results = engine.benchmark(model_name)
    print("Benchmark complete!\n")


def cmd_profile(args):
    """Handle profile command."""
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    profile = engine.profile(args.model)
    
    print("\nModel Requirements")
    print("=" * 50)
    for k, v in profile.items():
        print(f"  {k.replace('_', ' ').title()}: {v}")
    print()


def cmd_info():
    """Handle info command."""
    from airt.engine import AIRTEngine
    engine = AIRTEngine()
    
    info = engine.info()
    
    print("\nAIRT-Engine Information")
    print("=" * 50)
    print(f"  Version: {info['version']}")
    print(f"  CUDA Available: {info['hardware']['cuda_available']}")
    
    if info['hardware'].get('gpu_name'):
        print(f"  GPU: {info['hardware']['gpu_name']}")
        print(f"  VRAM: {info['hardware']['vram_gb']} GB")
    
    print("\n  Configuration:")
    for k, v in info['config'].items():
        print(f"    {k}: {v}")
    print()


def start_server(args):
    """Start the web server."""
    print(f"\nStarting AIRT-Engine server on http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    import uvicorn
    uvicorn.run(
        "server.api:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == '__main__':
    main()