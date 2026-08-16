import io
import torch
import torchaudio as ta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from chatterbox.tts import ChatterboxTTS
from contextlib import asynccontextmanager
import uvicorn

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Loading ChatterboxTTS on {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully.")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/generate_stream")
async def websocket_generate_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    try:
        while True:
            # Wait for JSON payload from client
            # Example: {"text": "Hello world", "exaggeration": 0.5, "chunk_size": 25}
            data = await websocket.receive_json()
            
            text = data.get("text")
            if not text:
                await websocket.send_json({"error": "Missing 'text' in payload"})
                continue
            
            exaggeration = data.get("exaggeration", 0.5)
            chunk_size = data.get("chunk_size", 25)
            cfg_weight = data.get("cfg_weight", 0.5)
            temperature = data.get("temperature", 0.8)
            
            print(f"Generating audio for text: '{text[:30]}...'")
            
            try:
                # generate_stream yields (audio_chunk, metrics)
                for audio_chunk, metrics in model.generate_stream(
                    text=text,
                    chunk_size=chunk_size,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    print_metrics=False
                ):
                    # Convert tensor to standalone .wav file bytes in memory
                    buffer = io.BytesIO()
                    ta.save(buffer, audio_chunk, model.sr, format="wav")
                    wav_bytes = buffer.getvalue()
                    
                    # Send the .wav bytes to the client
                    await websocket.send_bytes(wav_bytes)
                
                # Signal the end of generation for this prompt by sending an empty bytes message
                await websocket.send_bytes(b"")
                print("Finished streaming audio for request.")
                
            except Exception as e:
                print(f"Error during streaming generation: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
