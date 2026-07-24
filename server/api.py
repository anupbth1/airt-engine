"""AIRT-Engine API - FastAPI routes"""
import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import base64

from airt.engine import AIRTEngine
from airt.utils.logger import log

# Create engine (lazy init)
engine = None

def get_engine() -> AIRTEngine:
    global engine
    if engine is None:
        engine = AIRTEngine()
    return engine

# FastAPI app
app = FastAPI(
    title="AIRT-Engine API",
    description="Adaptive Intelligence Runtime Engine - Multi-modal AI",
    version="1.0.0",
)


# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int


class ImageAnalysisRequest(BaseModel):
    prompt: str = "Describe this image in detail."
    image_base64: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    from fastapi.templating import Jinja2Templates
    from fastapi.requests import Request
    
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
    
    # Simple HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIRT-Engine</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   background: #0f172a; color: #e2e8f0; min-height: 100vh; }
            .container { max-width: 900px; margin: 0 auto; padding: 20px; }
            h1 { color: #38bdf8; margin-bottom: 10px; font-size: 2em; }
            .subtitle { color: #94a3b8; margin-bottom: 30px; }
            .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px;
                     border: 1px solid #334155; }
            label { display: block; margin-bottom: 8px; color: #94a3b8; font-weight: 500; }
            textarea, input[type="text"], select { width: 100%; padding: 12px; border-radius: 8px;
                  border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 14px; }
            textarea { min-height: 100px; resize: vertical; }
            button { background: #38bdf8; color: #0f172a; border: none; padding: 12px 24px;
                     border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px; }
            button:hover { background: #7dd3fc; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            .response-box { background: #0f172a; border: 1px solid #334155; border-radius: 8px;
                           padding: 16px; margin-top: 16px; min-height: 50px; white-space: pre-wrap; }
            .tab-bar { display: flex; gap: 8px; margin-bottom: 16px; }
            .tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; background: #334155; }
            .tab.active { background: #38bdf8; color: #0f172a; }
            .file-input { padding: 40px; border: 2px dashed #334155; border-radius: 8px; 
                         text-align: center; cursor: pointer; margin-bottom: 16px; }
            .file-input:hover { border-color: #38bdf8; }
            .status { margin-top: 8px; color: #94a3b8; font-size: 0.9em; }
            .model-info { background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                         padding: 12px; margin-top: 16px; font-size: 0.9em; }
            .model-info span { color: #38bdf8; }
            .badge { display: inline-block; background: #334155; padding: 4px 8px; border-radius: 4px;
                    font-size: 0.8em; margin-right: 4px; }
            .flex-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AIRT-Engine</h1>
            <p class="subtitle">Adaptive Intelligence Runtime — Text, Image, Audio, Speech</p>
            
            <div class="tab-bar" id="tabBar">
                <div class="tab active" onclick="switchTab('chat')">💬 Chat</div>
                <div class="tab" onclick="switchTab('vision')">🖼️ Vision</div>
                <div class="tab" onclick="switchTab('audio')">🎤 Audio</div>
                <div class="tab" onclick="switchTab('tts')">🔊 TTS</div>
                <div class="tab" onclick="switchTab('benchmark')">📊 Benchmark</div>
            </div>
            
            <!-- Chat Tab -->
            <div id="tab-chat" class="tab-content">
                <div class="card">
                    <label for="chatPrompt">Enter your prompt</label>
                    <textarea id="chatPrompt" placeholder="Ask anything..."></textarea>
                    <div class="flex-row" style="margin-top:12px">
                        <button onclick="sendChat()" id="chatBtn">Send</button>
                        <select id="chatModel" style="width:auto;flex:1">
                            <option value="microsoft/Phi-3.5-mini-instruct">Phi-3.5-mini (3.8B)</option>
                            <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B</option>
                            <option value="meta-llama/Llama-3.2-1B-Instruct">Llama-3.2-1B</option>
                        </select>
                    </div>
                    <div id="chatResponse" class="response-box">Response will appear here...</div>
                    <div id="chatStatus" class="status"></div>
                </div>
            </div>
            
            <!-- Vision Tab -->
            <div id="tab-vision" class="tab-content" style="display:none">
                <div class="card">
                    <label>Upload an image</label>
                    <div class="file-input" onclick="document.getElementById('imageInput').click()">
                        <p>Click to upload image</p>
                        <p style="font-size:0.8em;color:#64748b;margin-top:8px">JPG, PNG, WEBP</p>
                    </div>
                    <input type="file" id="imageInput" accept="image/*" style="display:none" onchange="handleImageUpload(event)">
                    <div id="imagePreview" style="display:none;margin-bottom:12px">
                        <img id="previewImg" style="max-width:300px;max-height:300px;border-radius:8px">
                    </div>
                    <label for="visionPrompt">Question about the image</label>
                    <input type="text" id="visionPrompt" value="Describe this image in detail.">
                    <button onclick="sendVision()" style="margin-top:12px" id="visionBtn">Analyze</button>
                    <div id="visionResponse" class="response-box">Analysis will appear here...</div>
                    <div id="visionStatus" class="status"></div>
                </div>
            </div>
            
            <!-- Audio Tab -->
            <div id="tab-audio" class="tab-content" style="display:none">
                <div class="card">
                    <label>Upload audio file for transcription</label>
                    <div class="file-input" onclick="document.getElementById('audioInput').click()">
                        <p>Click to upload audio</p>
                        <p style="font-size:0.8em;color:#64748b;margin-top:8px">MP3, WAV, M4A</p>
                    </div>
                    <input type="file" id="audioInput" accept="audio/*" style="display:none" onchange="handleAudioUpload(event)">
                    <div id="audioFileName" class="status"></div>
                    <button onclick="sendAudio()" style="margin-top:12px" id="audioBtn">Transcribe</button>
                    <div id="audioResponse" class="response-box">Transcription will appear here...</div>
                    <div id="audioStatus" class="status"></div>
                </div>
            </div>
            
            <!-- TTS Tab -->
            <div id="tab-tts" class="tab-content" style="display:none">
                <div class="card">
                    <label for="ttsText">Text to speak</label>
                    <textarea id="ttsText" placeholder="Enter text to convert to speech..."></textarea>
                    <div class="flex-row" style="margin-top:12px">
                        <button onclick="sendTTS()" id="ttsBtn">Generate Speech</button>
                        <select id="ttsLang" style="width:auto">
                            <option value="en">English</option>
                            <option value="hi">Hindi</option>
                            <option value="zh">Chinese</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                        </select>
                    </div>
                    <div id="ttsResponse" class="response-box">Audio will appear here...</div>
                    <div id="ttsStatus" class="status"></div>
                </div>
            </div>
            
            <!-- Benchmark Tab -->
            <div id="tab-benchmark" class="tab-content" style="display:none">
                <div class="card">
                    <label>Benchmark a model</label>
                    <div class="flex-row" style="margin-bottom:12px">
                        <select id="benchModel" style="flex:1">
                            <option value="microsoft/Phi-3.5-mini-instruct">Phi-3.5-mini (3.8B)</option>
                            <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B</option>
                            <option value="meta-llama/Llama-3.2-1B-Instruct">Llama-3.2-1B</option>
                        </select>
                        <button onclick="runBenchmark()" id="benchBtn">Run Benchmark</button>
                    </div>
                    <div id="benchResponse" class="response-box">Results will appear here...</div>
                    <div id="benchStatus" class="status"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Tab switching
            function switchTab(name) {
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.getElementById('tab-' + name).style.display = 'block';
                event.target.classList.add('active');
            }
            
            // Chat
            async function sendChat() {
                const prompt = document.getElementById('chatPrompt').value;
                if (!prompt) return;
                const btn = document.getElementById('chatBtn');
                const status = document.getElementById('chatStatus');
                btn.disabled = true;
                status.textContent = 'Generating...';
                
                try {
                    const res = await fetch('/api/v1/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: prompt,
                            model_name: document.getElementById('chatModel').value,
                            max_tokens: 256,
                            temperature: 0.7
                        })
                    });
                    const data = await res.json();
                    document.getElementById('chatResponse').textContent = data.response;
                    status.textContent = `Model: ${data.model_used} | Tokens: ${data.tokens_generated}`;
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
                btn.disabled = false;
            }
            
            // Vision
            let uploadedImage = null;
            function handleImageUpload(e) {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(ev) {
                    document.getElementById('previewImg').src = ev.target.result;
                    document.getElementById('imagePreview').style.display = 'block';
                    uploadedImage = ev.target.result.split(',')[1];
                };
                reader.readAsDataURL(file);
            }
            
            async function sendVision() {
                const btn = document.getElementById('visionBtn');
                const status = document.getElementById('visionStatus');
                if (!uploadedImage) { status.textContent = 'Please upload an image first.'; return; }
                btn.disabled = true;
                status.textContent = 'Analyzing...';
                
                try {
                    const res = await fetch('/api/v1/analyze-image', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: document.getElementById('visionPrompt').value,
                            image_base64: uploadedImage
                        })
                    });
                    const data = await res.json();
                    document.getElementById('visionResponse').textContent = data.analysis;
                    status.textContent = 'Complete';
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
                btn.disabled = false;
            }
            
            // Audio
            let uploadedAudio = null;
            function handleAudioUpload(e) {
                const file = e.target.files[0];
                if (!file) return;
                uploadedAudio = file;
                document.getElementById('audioFileName').textContent = 'File: ' + file.name;
            }
            
            async function sendAudio() {
                const btn = document.getElementById('audioBtn');
                const status = document.getElementById('audioStatus');
                if (!uploadedAudio) { status.textContent = 'Please upload an audio file first.'; return; }
                btn.disabled = true;
                status.textContent = 'Transcribing...';
                
                const formData = new FormData();
                formData.append('file', uploadedAudio);
                
                try {
                    const res = await fetch('/api/v1/transcribe', { method: 'POST', body: formData });
                    const data = await res.json();
                    document.getElementById('audioResponse').textContent = data.transcription;
                    status.textContent = 'Complete';
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
                btn.disabled = false;
            }
            
            // TTS
            async function sendTTS() {
                const text = document.getElementById('ttsText').value;
                if (!text) return;
                const btn = document.getElementById('ttsBtn');
                const status = document.getElementById('ttsStatus');
                btn.disabled = true;
                status.textContent = 'Generating speech...';
                
                try {
                    const res = await fetch('/api/v1/tts', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: text,
                            lang: document.getElementById('ttsLang').value
                        })
                    });
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    document.getElementById('ttsResponse').innerHTML = 
                        '<audio controls src="' + url + '" style="width:100%"></audio>';
                    status.textContent = 'Complete';
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
                btn.disabled = false;
            }
            
            // Benchmark
            async function runBenchmark() {
                const btn = document.getElementById('benchBtn');
                const status = document.getElementById('benchStatus');
                btn.disabled = true;
                status.textContent = 'Running benchmark (this may take a while)...';
                
                try {
                    const res = await fetch('/api/v1/benchmark', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            model_name: document.getElementById('benchModel').value
                        })
                    });
                    const data = await res.json();
                    let html = '<pre style="font-family:monospace">';
                    html += '='.repeat(50) + '\\n';
                    html += '  Model: ' + data.summary.model + '\\n';
                    html += '  Accuracy: ' + data.summary.accuracy_pct + '%\\n';
                    html += '  Speed: ' + data.summary.tokens_per_sec + ' tokens/sec\\n';
                    html += '  Latency: ' + data.summary.latency_s + 's\\n';
                    html += '  Memory: ' + data.summary.memory_gb + ' GB\\n';
                    html += '='.repeat(50) + '\\n';
                    html += '</pre>';
                    document.getElementById('benchResponse').innerHTML = html;
                    status.textContent = 'Complete';
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
                btn.disabled = false;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# API Routes
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI."""
    eng = get_engine()
    response = eng.chat(
        request.prompt,
        model_name=request.model_name,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    tokens = len(response.split())
    return ChatResponse(
        response=response,
        model_used=request.model_name or eng.loader.detect_best_backend(),
        tokens_generated=tokens,
    )


@app.post("/api/v1/analyze-image")
async def analyze_image_endpoint(request: ImageAnalysisRequest):
    """Analyze an image."""
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Save base64 image to temp file
    import tempfile
    image_data = base64.b64decode(request.image_base64)
    suffix = '.png'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name
    
    try:
        eng = get_engine()
        analysis = eng.analyze_image(tmp_path, request.prompt)
        return {"analysis": analysis}
    finally:
        os.unlink(tmp_path)


@app.post("/api/v1/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Transcribe audio to text."""
    suffix = os.path.splitext(file.filename or '.mp3')[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        eng = get_engine()
        transcription = eng.transcribe(tmp_path)
        return {"transcription": transcription}
    finally:
        os.unlink(tmp_path)


@app.post("/api/v1/tts")
async def tts_endpoint(request: TTSRequest):
    """Convert text to speech."""
    eng = get_engine()
    audio_bytes = eng.speak(request.text, request.lang)
    
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=speech.mp3"}
    )


@app.post("/api/v1/benchmark")
async def benchmark_endpoint(request: ChatRequest):
    """Run benchmark on a model."""
    eng = get_engine()
    results = eng.benchmark(request.model_name)
    return results


@app.get("/api/v1/info")
async def info_endpoint():
    """Get engine information."""
    eng = get_engine()
    return eng.info()