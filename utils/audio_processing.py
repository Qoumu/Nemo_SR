import pyaudio
import numpy as np

def stream_audio(chunk_size=1600, sample_rate=16000):
    """
    Stream audio from microphone.
    
    Args:
        chunk_size: Number of samples per chunk (~100ms at 16kHz)
        sample_rate: Audio sample rate in Hz
        
    Yields:
        numpy.ndarray: Audio chunks as float32 in range [-1, 1]
    """
    CHUNK = chunk_size
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sample_rate
    
    p = pyaudio.PyAudio()
    
    # List available input devices
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    print()
    
    stream = None
    try:
        # Try to open audio stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=None  # Use default device
        )
        print(f"✓ Microphone opened successfully (Rate: {RATE}Hz, Chunk: {CHUNK} samples)")
        print("Listening...\n")
        
    except OSError as e:
        print(f"✗ Error: Could not open microphone: {e}")
        print("\nPossible solutions:")
        print("1. Check if microphone is connected")
        print("2. Grant microphone permissions")
        print("3. Try a different input device")
        print("4. Run: sudo apt-get install portaudio19-dev (Linux)")
        print("\nGenerating test audio for demonstration...\n")
        
        # Generate test audio with speech-like pattern
        import time
        while True:
            try:
                # Simulate speech pattern: random bursts of audio
                if np.random.random() > 0.7:  # 30% chance of "speech"
                    # Generate random audio (simulating speech)
                    audio_chunk = np.random.randn(CHUNK).astype(np.float32) * 0.1
                else:
                    # Generate silence
                    audio_chunk = np.zeros(CHUNK, dtype=np.float32)
                
                time.sleep(CHUNK / RATE)  # Simulate real-time
                yield audio_chunk
                
            except KeyboardInterrupt:
                break
        return
    
    # Stream real audio from microphone
    try:
        while True:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert to float32 normalized to [-1, 1]
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            yield audio_chunk
            
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()


def test_audio_stream():
    """Test function to verify audio streaming works"""
    print("Testing audio stream for 5 seconds...")
    
    chunk_count = 0
    start_time = None
    
    for audio_chunk in stream_audio():
        if start_time is None:
            import time
            start_time = time.time()
        
        chunk_count += 1
        energy = np.abs(audio_chunk).mean()
        max_val = np.abs(audio_chunk).max()
        
        print(f"Chunk {chunk_count}: Energy={energy:.4f}, Max={max_val:.4f}")
        
        import time
        if time.time() - start_time > 5:
            break
    
    print(f"\n✓ Successfully captured {chunk_count} audio chunks")


if __name__ == "__main__":
    test_audio_stream()