import time
import random
import asyncio
import websockets
import argparse
import json
import io
import scipy.io.wavfile
import numpy as np

def generate_physics_essays(count=50):
    intros = [
        "Have you ever wondered why the universe behaves the way it does? This is one of the most profound questions in modern physics. ",
        "When we look up at the night sky, we are essentially looking back in time, witnessing the ancient history of the cosmos. ",
        "The fundamental principles of nature are hidden behind mathematical equations that describe everything from tiny particles to massive galaxies. "
    ]
    concepts = [
        "In the realm of classical mechanics, everything seems predictable, like a giant clockwork mechanism running smoothly in the background. ",
        "However, when we delve into quantum mechanics, particles can exist in multiple places simultaneously, a bizarre phenomenon known as superposition. ",
        "Albert Einstein revolutionized our understanding with his theory of relativity, showing that space and time are fundamentally woven together. "
    ]
    elaborations = [
        "This means that at the fundamental level, reality is not deterministic but governed by probabilities and complex wave functions. ",
        "Massive objects like black holes warp this spacetime fabric so intensely that not even light can escape their immense gravitational pull. ",
        "This relentless increase in disorder dictates the arrow of time, locking us into a journey moving strictly forward into the future. "
    ]
    details = [
        "Think of the famous double-slit experiment, where mere observation changes the outcome, suggesting that consciousness might play a role in reality. ",
        "Consider the cosmic microwave background radiation, an ancient echo of the Big Bang that still permeates every single inch of empty space. ",
        "Imagine the Large Hadron Collider, smashing subatomic particles together at nearly the speed of light to uncover the fundamental building blocks. "
    ]
    conclusions = [
        "Ultimately, these phenomena prove that the universe is far stranger, and far more beautiful, than we could have ever possibly imagined. ",
        "As we continue to develop better mathematical models, we edge ever closer to discovering a unified theory of everything that governs reality. ",
        "The true beauty of physics lies in its ability to reduce such unimaginably complex systems into elegant, simple, and universal mathematical equations. "
    ]
    padding = [
        "Every new discovery simply opens the door to even more profound and perplexing questions about our existence.",
        "We are, after all, just a way for the cosmos to consciously think about and understand its own true nature.",
        "The pursuit of this knowledge requires immense patience, rigorous mathematics, and a profound sense of unending human curiosity."
    ]

    essays = []
    for _ in range(count):
        c1, c2 = random.sample(concepts, 2)
        e1, e2 = random.sample(elaborations, 2)
        
        essay = (
            random.choice(intros) + c1 + e1 + c2 + e2 +
            random.choice(details) + random.choice(conclusions) + random.choice(padding)
        )
        essays.append(f"Test sequence {random.randint(10000, 99999)}. {essay}")
    
    return essays

async def perform_request(url, req_id, text_payload, diffusion_steps):
    payload = {
        "text": text_payload,
        "diffusion_steps": diffusion_steps,
        "chunk_size": 25,
        "exaggeration": 0.5
    }
    
    start_time = time.time()
    latency = None
    total_audio_duration = 0.0
    all_audio_data = []
    final_sr = None
    
    try:
        async with websockets.connect(url, ping_interval=None) as websocket:
            await websocket.send(json.dumps(payload))
            
            while True:
                message = await websocket.recv()
                
                if not message:
                    break
                
                if isinstance(message, bytes):
                    if latency is None:
                        latency = time.time() - start_time
                        
                    try:
                        with io.BytesIO(message) as f:
                            sr, audio_data = scipy.io.wavfile.read(f)
                            
                        num_samples = len(audio_data) if len(audio_data.shape) == 1 else audio_data.shape[0]
                        total_audio_duration += num_samples / sr
                        
                        all_audio_data.append(audio_data)
                        final_sr = sr
                    except Exception as e:
                        print(f"[Req {req_id}] Audio decode error: {str(e)}")
                        return None
                
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if "error" in data:
                            print(f"[Req {req_id}] Server Error: {data['error']}")
                            return None
                    except json.JSONDecodeError:
                        pass
                        
            delivery_time = time.time() - start_time
            
            if total_audio_duration == 0:
                print(f"[Req {req_id}] No audio received")
                return None
                
            if all_audio_data and final_sr:
                combined_audio = np.concatenate(all_audio_data)
                safe_req_id = str(req_id).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
                filename = f"physics_req_{safe_req_id}.wav"
                scipy.io.wavfile.write(filename, final_sr, combined_audio)
                
            rtf = delivery_time / total_audio_duration if total_audio_duration > 0 else 0
            
            print(f"  [Req {req_id}] Success | TTFA: {latency:.2f}s | Delivery: {delivery_time:.2f}s | Audio: {total_audio_duration:.2f}s | RTF: {rtf:.2f}x")
            
            return {
                "req_id": req_id,
                "latency": latency,
                "delivery_time": delivery_time,
                "audio_duration": total_audio_duration,
                "rtf": rtf
            }
            
    except Exception as e:
        print(f"[Req {req_id}] Exception: {str(e)}")
        return None

async def traffic_generator(url, essays_pool, diffusion_steps, duration, arrival_rate, phase_name):
    """
    Spawns requests at an average arrival_rate (requests per second) for a given duration.
    Uses an exponential distribution for inter-arrival times to simulate real-world, unpredictable traffic.
    """
    print(f"\n{'='*50}")
    print(f"--- Starting Phase: {phase_name} | Target: {arrival_rate} req/s for {duration} seconds ---")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    tasks = []
    req_count = 0
    
    while time.time() - start_time < duration:
        if not essays_pool:
            print("Out of essays! Stopping generation.")
            break
            
        req_count += 1
        req_id = f"{phase_name}_{req_count}"
        text_payload = essays_pool.pop(0)
        
        # Fire and forget the request
        tasks.append(asyncio.create_task(perform_request(url, req_id, text_payload, diffusion_steps)))
        
        # Calculate random delay to next request using exponential distribution (Poisson process)
        # E.g., if arrival_rate is 2 req/s, average delay is 0.5s, but highly variable.
        delay = random.expovariate(arrival_rate)
        await asyncio.sleep(delay)
        
    print(f"Phase '{phase_name}' generated {req_count} requests.")
    return tasks

async def main():
    parser = argparse.ArgumentParser(description="Variable Real-World Load Testing Script")
    parser.add_argument("--url", type=str, default="ws://10.9.3.166:8000/ws/generate_stream", help="Target WebSocket URL")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps (may be ignored by the backend)")
    args = parser.parse_args()
    
    # Define traffic phases using Requests Per Minute (converted to req/second for the math).
    # This keeps the total volume similar to your 10 and 20 concurrency script (around 30-50 total requests),
    # but spaces them out variably to prevent instant OOM crashes.
    phases = [
        {"name": "Warmup (2 req/min)", "duration": 300, "rate": 2.0 / 60.0},     # 1 request every 30s for 5 mins
        {"name": "Steady Load (10 req/min)", "duration": 900, "rate": 10.0 / 60.0}, # 1 request every 6s for 15 mins
        {"name": "Peak Load (25 req/min)", "duration": 300, "rate": 25.0 / 60.0},   # 1 request every 2.4s for 5 mins
        {"name": "Cooldown (5 req/min)", "duration": 300, "rate": 5.0 / 60.0}      # 1 request every 12s for 5 mins
    ]
    
    # Calculate how many essays we need roughly + buffer
    total_expected = sum(int(p["duration"] * p["rate"]) for p in phases) + 50 
    
    print(f"Generating {total_expected} unique physics essays for variable load testing...")
    essays_pool = generate_physics_essays(total_expected)
    print("Essays generated successfully!\n")
    
    print(f"Starting variable real-world load test on {args.url}")
    
    all_tasks = []
    
    # Execute the traffic schedule
    for phase in phases:
        tasks = await traffic_generator(args.url, essays_pool, args.steps, phase["duration"], phase["rate"], phase["name"])
        all_tasks.extend(tasks)
        
    print("\nAll traffic generated! Waiting for active requests to finish processing...")
    results = await asyncio.gather(*all_tasks)
        
    successful = [r for r in results if r is not None]
    
    print(f"\n{'='*50}")
    print("FINAL LOAD TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Total Requests Spawned : {len(all_tasks)}")
    print(f"  Successful Requests    : {len(successful)}")
    print(f"  Failed Requests        : {len(all_tasks) - len(successful)}")
    
    if successful:
        avg_latency = sum(r["latency"] for r in successful) / len(successful)
        avg_delivery = sum(r["delivery_time"] for r in successful) / len(successful)
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        avg_dur = sum(r["audio_duration"] for r in successful) / len(successful)
        
        print(f"  Avg Audio Dur          : {avg_dur:.2f} s")
        print(f"  Avg TTFA               : {avg_latency:.2f} s")
        print(f"  Avg Delivery Time      : {avg_delivery:.2f} s")
        print(f"  Avg RTF                : {avg_rtf:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
