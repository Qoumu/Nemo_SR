import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import numpy as np
from utils.audio_processing import stream_audio

# Load model
model = nemo_asr.models.EncDecRNNTModel.restore_from(
    "models/stt_en_conformer_transducer_medium.nemo"
)

# Disable CUDA graphs
decoding_cfg = OmegaConf.create({
    'strategy': 'greedy_batch',
    'greedy': {'use_cuda_graph_decoder': False}
})
model.change_decoding_strategy(decoding_cfg)
model.eval()

BUFFER_SIZE = int(2.0 * 16000)  # 2 seconds
audio_buffer = np.array([], dtype=np.float32)

print("Listening... (Speak into microphone)")

for audio_chunk in stream_audio():
    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
    
    if len(audio_buffer) >= BUFFER_SIZE:
        audio_to_process = audio_buffer[:BUFFER_SIZE]
        
        # Check for actual audio (not silence)
        if np.abs(audio_to_process).mean() > 0.001:
            # Transcribe using correct API
            hypotheses = model.transcribe(audio=[audio_to_process], batch_size=1)
            
            if hypotheses and hypotheses[0]:
                text = hypotheses[0].text if hasattr(hypotheses[0], 'text') else str(hypotheses[0])
                if text.strip():
                    print(f"→ {text}")
        
        # Keep 0.5s overlap
        audio_buffer = audio_buffer[int(1.5 * 16000):]