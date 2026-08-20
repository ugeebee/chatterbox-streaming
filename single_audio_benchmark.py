import asyncio
import websockets
import json
import time
import io
import scipy.io.wavfile
import numpy as np

async def test_stream():
    uri = "ws://10.9.3.166:8000/ws/generate_stream"
    payload = {
        "text": "Hello, this is a real-time streaming test of the Chatterbox Text-to-Speech model via WebSockets. I am generating just one audio to benchmark TTFA.",
        "chunk_size": 25,
        "exaggeration": 0.5
    }
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, ping_interval=None) as websocket:
            print("Connected! Sending payload...")
            start_time = time.time()
            await websocket.send(json.dumps(payload))
            
            ttfa = None
            audio_chunks = []
            final_sr = None
            
            while True:
                message = await websocket.recv()
                
                if not message:
                    print("Received end of stream signal.")
                    break
                    
                if isinstance(message, bytes):
                    if ttfa is None:
                        ttfa = time.time() - start_time
                        print(f"\n🚀 TTFA (Time to First Audio): {ttfa:.4f} seconds\n")
                    
                    with io.BytesIO(message) as f:
                        sr, audio_data = scipy.io.wavfile.read(f)
                        audio_chunks.append(audio_data)
                        final_sr = sr
                        
                elif isinstance(message, str):
                    print(f"Server message: {message}")
                    
            if audio_chunks and final_sr:
                combined_audio = np.concatenate(audio_chunks)
                filename = "single_benchmark_output.wav"
                scipy.io.wavfile.write(filename, final_sr, combined_audio)
                
                # Calculate audio duration
                num_samples = len(combined_audio) if len(combined_audio.shape) == 1 else combined_audio.shape[0]
                audio_duration = num_samples / final_sr
                
                total_time = time.time() - start_time
                print(f"Finished! Saved to {filename}")
                print(f"Audio Duration: {audio_duration:.2f} seconds")
                print(f"Total delivery time: {total_time:.4f} seconds")
                if audio_duration > 0:
                    print(f"RTF (Real-Time Factor): {total_time/audio_duration:.2f}x")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
