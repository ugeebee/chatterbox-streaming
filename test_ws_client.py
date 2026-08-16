import asyncio
import websockets
import json

async def test_stream():
    uri = "ws://localhost:8000/ws/generate_stream"
    
    # Payload for the TTS request
    payload = {
        "text": "Hello, this is a real-time streaming test of the Chatterbox Text-to-Speech model via WebSockets.",
        "chunk_size": 25,
        "exaggeration": 0.5
    }
    
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("Connected! Sending payload...")
        await websocket.send(json.dumps(payload))
        
        chunk_count = 0
        while True:
            try:
                # Receive message
                message = await websocket.recv()
                
                # Check if it's the end of stream signal
                if not message:
                    print("Received end of stream signal.")
                    break
                    
                # If it's bytes, it's a wav chunk
                if isinstance(message, bytes):
                    chunk_count += 1
                    filename = f"test_chunk_{chunk_count}.wav"
                    with open(filename, "wb") as f:
                        f.write(message)
                    print(f"Received chunk {chunk_count}, saved to {filename} ({len(message)} bytes)")
                
                # If it's a string, it might be an error JSON
                elif isinstance(message, str):
                    print(f"Received JSON message: {message}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                break

if __name__ == "__main__":
    asyncio.run(test_stream())
