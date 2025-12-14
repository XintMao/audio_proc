import sys
import torch
import whisperx
from pyannote.audio import Pipeline
import json
import os
from pyannote.core import Segment, Annotation
import torchaudio
import time
from transformers import AutoTokenizer
import librosa

# Required imports for PyTorch 2.6+ safe loading (fix WeightsUnpickler error)
import torch.serialization 
import omegaconf.listconfig 
import omegaconf.base
import omegaconf.nodes
import typing
import collections
import omegaconf.dictconfig
import torch.torch_version
import pyannote.audio.core.model
import pyannote.audio.core.task

# Enable safe globals for Pyannote model loading
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        # Omegaconf dependencies
        omegaconf.listconfig.ListConfig, 
        omegaconf.base.ContainerMetadata,
        omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata,
        omegaconf.dictconfig.DictConfig,
        
        # Pyannote core fixes
        torch.torch_version.TorchVersion, 
        pyannote.audio.core.model.Introspection,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution,
        
        # Python built-in types
        list,
        dict,
        int,
        typing.Any,
        collections.defaultdict,
    ])
    print("Enabled core safe globals for Pyannote model loading.")

# Configuration
BASE_FILENAME = "01281_ARM" 
AUDIO_PATH = f"./data/{BASE_FILENAME}.mp3" 
TRANSCRIPT_PATH = f"./data/{BASE_FILENAME}.txt"
OUTPUT_DIR = "./output"
LANGUAGE = "hy"  # Armenian
MAX_SEGMENT_DURATION = 30.0  # Max segment length (seconds)

# Create output dir if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Configuration ---")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Target language: {LANGUAGE}")

# GPU config (optimized for RTX 4090D)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 
compute_type = "float16" if torch.cuda.is_available() else "float32"

# -------------------------------------------------------------
# Core segment splitting and formatting functions
# -------------------------------------------------------------

def split_segments_by_length(aligned_data, max_duration=MAX_SEGMENT_DURATION):
    """Split aligned segments by max duration, cutting at word boundaries"""
    all_words = []
    for segment in aligned_data.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    if not all_words:
        return {'segments': []}

    # Generate (start, end) boundaries
    chunks = []
    current_chunk_start = all_words[0]['start']
    prev_word = None

    for word in all_words:
        word_end = word["end"]
        if word_end - current_chunk_start > max_duration:
            if prev_word is not None:
                chunks.append((current_chunk_start, prev_word["end"]))
                current_chunk_start = prev_word["end"]
            prev_word = word
        prev_word = word

    # Add final chunk
    if prev_word is not None and (not chunks or chunks[-1][1] != prev_word["end"]):
         chunks.append((current_chunk_start, prev_word["end"]))
    
    # Rebuild segments from boundaries
    new_segments = []
    for start_time, end_time in chunks:
        current_chunk_words = [w for w in all_words if w['start'] >= start_time and w['end'] <= end_time]
        
        if current_chunk_words:
            new_text = " ".join([w['word'] for w in current_chunk_words])
            new_segments.append({
                'start': start_time,
                'end': end_time,
                'text': new_text,
                'words': current_chunk_words
            })
            
    return {'segments': new_segments}

def format_time_srt(seconds):
    """Convert seconds to HH:MM:SS,mmm SRT format"""
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds / 60) % 60
    h = int(seconds / 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" 

def merge_and_format(alignment_path, rttm_path):
    """Merge time-aligned segments with speaker diarization results"""
    with open(alignment_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)
    alignment_segments = aligned_data.get('segments', []) 

    diarization_annotation = Annotation(uri="audio_file", modality="speaker")
    with open(rttm_path, 'r') as f:
        for line in f:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
                speaker_id = parts[7]
                diarization_annotation[Segment(start, end)] = speaker_id
    
    final_output = []
    
    for segment in alignment_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', seg_start)
        text = segment.get('text', '').strip()
        
        if not text: continue
            
        current_segment = Segment(seg_start, seg_end)
        overlap = diarization_annotation.crop(current_segment)
        
        speaker = 'UNKNOWN'
        if overlap:
            max_duration = 0
            for s, _, spk in overlap.itertracks(yield_label=True):
                duration = s.duration
                if duration > max_duration:
                    max_duration = duration
                    speaker = spk
            
            if speaker != 'UNKNOWN':
                speaker_id_num = int(speaker.split('_')[-1]) + 1  # Convert 0-based to 1-based
                speaker_label = f"[Speaker {speaker_id_num}]" 
            else:
                speaker_label = '[UNKNOWN]'
        else:
            speaker_label = '[UNKNOWN]'

        start_str = format_time_srt(seg_start)
        end_str = format_time_srt(seg_end)
        
        output_line = f"{start_str} --> {end_str} {speaker_label}\n{text}\n\n"
        final_output.append(output_line)
        
    return final_output

# -------------------------------------------------------------
# 1. Run WhisperX Forced Alignment
# -------------------------------------------------------------
print("\n--- 1. Running WhisperX Forced Alignment ---")

# Output file paths
alignment_json_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_alignment.json")
resegmented_json_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_resegmented_alignment.json")
rttm_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_diarization.rttm")

try:
    # Load Whisper ASR model
    print("Loading Whisper ASR model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio_data, sr = librosa.load(AUDIO_PATH, sr=16000)
    audio = audio_data

    # Critical fix: Get audio duration via torchaudio
    metadata_info = torchaudio.info(AUDIO_PATH) 
    audio_duration = metadata_info.num_frames / metadata_info.sample_rate 
    print(f"Audio total duration: {audio_duration:.2f} seconds")
    
    # Armenian alignment model ID
    ARMENIAN_ALIGN_MODEL_ID = "facebook/wav2vec2-base"
    
    # Load Wav2Vec2 alignment model
    print("Loading Wav2Vec2 alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=LANGUAGE, 
        model_name=ARMENIAN_ALIGN_MODEL_ID, 
        device=device
    )

    # Critical fix: Manually load tokenizer and wrap in dict
    align_tokenizer = AutoTokenizer.from_pretrained(ARMENIAN_ALIGN_MODEL_ID)
    print("Successfully loaded tokenizer manually.")
    align_model_meta = {'tokenizer': align_tokenizer}  # Fix 'object is not subscriptable' error
    
    # Read transcript for alignment
    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        text_to_align = f.read().strip()

    # Build required input format for whisperx.align
    print("Constructing alignment input format...")
    align_input = {
        'segments': [
            {
                'text': text_to_align,
                'start': 0,      # Required for forced alignment
                'end': audio_duration  # Use actual audio duration
            }
        ],
        'language': LANGUAGE  # Mandatory language info
    }

    # Run forced alignment
    print("Starting forced alignment...")
    start_time_align = time.time()
    result_aligned = whisperx.align(
        align_input, 
        model_a, 
        align_model_meta,  # Pass wrapped tokenizer dict
        audio, 
        device
    ) 
    end_time_align = time.time()
    print(f"Forced alignment completed in {end_time_align - start_time_align:.2f} seconds")

    # Save raw alignment results (word-level timestamps)
    with open(alignment_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=4)
    print(f"Raw alignment results saved to: {alignment_json_path}")

    # -------------------------------------------------------------
    # 1.5. Resegment by max duration
    # -------------------------------------------------------------
    print(f"\n--- 1.5. Resegmenting by max duration ({MAX_SEGMENT_DURATION}s) ---")
    
    # Load full alignment results
    with open(alignment_json_path, 'r', encoding='utf-8') as f:
        full_aligned_result = json.load(f)

    # Run resegmentation
    resegmented_result = split_segments_by_length(full_aligned_result)
    
    # Save resegmented results
    with open(resegmented_json_path, 'w', encoding='utf-8') as f:
        json.dump(resegmented_result, f, ensure_ascii=False, indent=4)
    print(f"Resegmented results saved to: {resegmented_json_path}")
    
except Exception as e:
    print(f"!!! WhisperX execution failed.")
    print(f"Error details: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 2. Run Pyannote Speaker Diarization
# -------------------------------------------------------------
print("\n--- 2. Running Pyannote Speaker Diarization ---")

try:
    # Load pyannote/speaker-diarization model
    print("Loading Pyannote Diarization model...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization") 
    diarization_pipeline.to(torch.device(device)) 

    # Run diarization
    print("Starting speaker diarization...")
    start_time_diar = time.time()
    diarization_result = diarization_pipeline(AUDIO_PATH)
    end_time_diar = time.time()
    print(f"Speaker diarization completed in {end_time_diar - start_time_diar:.2f} seconds")

    # Save diarization results (RTTM format)
    with open(rttm_path, "w") as f:
        diarization_result.write_rttm(f)
    print(f"Speaker diarization results saved to: {rttm_path}")
    
except Exception as e:
    print(f"!!! Pyannote Diarization failed. Ensure `huggingface-cli login` is completed.")
    print(f"Error details: {e}")
    sys.exit(1)

# -------------------------------------------------------------
# 4. Merge results and save final output
# -------------------------------------------------------------
try:
    # Use resegmented JSON as input for merging
    combined_results = merge_and_format(resegmented_json_path, rttm_path) 

    # Save final formatted transcript
    final_output_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_final_transcript_formatted.txt")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        f.writelines(combined_results)

    print(f"\n✅ Task completed! Final formatted transcript saved to: {final_output_path}")

except Exception as e:
    print(f"!!! Final result merging failed.")
    print(f"Error details: {e}")
    sys.exit(1)
