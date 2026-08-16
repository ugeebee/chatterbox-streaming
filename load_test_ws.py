import asyncio
import websockets
import time
import json
import io
import argparse
import random
import scipy.io.wavfile

# A ~350 word text block to serve as the narration base.
# It is prefixed with a random number in the actual payload to bypass vLLM prefix caching.
BASE_TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice without pictures or conversation? So she was considering in her own mind as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late! when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural; but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge. In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs.
"""

async def perform_request(url, save_path=None):
    # Prepend a random number to break any prefix caching.
    unique_text = f"Test sequence {random.randint(10000, 99999)}. " + BASE_TEXT.strip()
    payload = {
        "text": unique_text,
        "chunk_size": 25,
        "exaggeration": 0.5
    }
    
    start_time = time.time()
    latency = None
    total_audio_duration = 0.0
    all_audio_data = []
    final_sr = None
    
    try:
        # Increase ping_interval to avoid timeouts on long generations
        async with websockets.connect(url, ping_interval=None) as websocket:
            await websocket.send(json.dumps(payload))
            
            while True:
                message = await websocket.recv()
                
                # Our API sends an empty bytes message to signal the end of the stream
                if not message:
                    break
                
                if isinstance(message, bytes):
                    # Record TTFB (Time To First Byte) on the first audio chunk received
                    if latency is None:
                        latency = time.time() - start_time
                        
                    # Decode audio chunk to get its duration
                    try:
                        with io.BytesIO(message) as f:
                            sr, audio_data = scipy.io.wavfile.read(f)
                            
                        # For 1D array (mono)
                        num_samples = len(audio_data) if len(audio_data.shape) == 1 else audio_data.shape[0]
                        total_audio_duration += num_samples / sr
                        
                        all_audio_data.append(audio_data)
                        final_sr = sr
                    except Exception as e:
                        return {"error": f"Audio decode error: {str(e)}"}
                
                elif isinstance(message, str):
                    # Check if the server sent a JSON error string
                    try:
                        data = json.loads(message)
                        if "error" in data:
                            return {"error": f"Server Error: {data['error']}"}
                    except json.JSONDecodeError:
                        pass
                        
            delivery_time = time.time() - start_time
            
            if total_audio_duration == 0:
                return {"error": "No audio received"}
                
            if save_path and all_audio_data and final_sr:
                import numpy as np
                combined_audio = np.concatenate(all_audio_data)
                scipy.io.wavfile.write(save_path, final_sr, combined_audio)
                
            rtf = delivery_time / total_audio_duration if total_audio_duration > 0 else 0
            
            return {
                "latency": latency,
                "delivery_time": delivery_time,
                "audio_duration": total_audio_duration,
                "rtf": rtf,
            }
            
    except Exception as e:
        return {"error": str(e)}

async def run_tier(concurrency, url):
    print(f"--- Running Load Tier: {concurrency} Concurrent Connections ---")
    
    tasks = []
    for i in range(concurrency):
        save_path = f"sample_tier_{concurrency}.wav" if i == 0 else None
        tasks.append(asyncio.create_task(perform_request(url, save_path)))
        
    # Wait for all tasks in this tier
    results = await asyncio.gather(*tasks)
        
    # Aggregate results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    if len(successful) > 0:
        avg_latency = sum(r["latency"] for r in successful) / len(successful)
        avg_delivery = sum(r["delivery_time"] for r in successful) / len(successful)
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        avg_dur = sum(r["audio_duration"] for r in successful) / len(successful)
        
        print(f"Results for Tier {concurrency}:")
        print(f"  Successful Requests : {len(successful)}")
        print(f"  Failed Requests     : {len(failed)}")
        print(f"  Avg Audio Duration  : {avg_dur:.2f} s")
        print(f"  Avg Latency (TTFB)  : {avg_latency:.2f} s")
        print(f"  Avg Delivery Time   : {avg_delivery:.2f} s")
        print(f"  Avg Real Time Factor: {avg_rtf:.2f}x")
    else:
        print(f"All {len(failed)} requests failed for tier {concurrency}.")
        if failed:
            print(f"Sample error: {failed[0]['error']}")
            
    print("\n")

async def main():
    parser = argparse.ArgumentParser(description="WebSocket Load testing script for Chatterbox Streaming Backend")
    parser.add_argument("--url", type=str, default="ws://10.9.3.166:8000/ws/generate_stream", help="Target WebSocket URL")
    args = parser.parse_args()
    
    tiers = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    print(f"Starting load test on {args.url}")
    print(f"Tiers: {tiers}")
    print("="*50)
    
    for concurrency in tiers:
        await run_tier(concurrency, args.url)
        # Give the server a small breather between tiers to flush memory
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
