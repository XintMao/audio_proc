import torch
import whisperx
from pyannote.audio import Pipeline
import json
import os
import sys
import numpy as np
import torchaudio
from pyannote.core import Segment, Annotation

# --- CONFIGURATION ---
BASE_FILENAME = "01_kapital_29_09_25"
AUDIO_PATH = f"./data/{BASE_FILENAME}.mp3"
TRANSCRIPT_PATH = f"./data/{BASE_FILENAME}.txt"
OUTPUT_DIR = "./output"
LANGUAGE = "hy"

# --- V77 HYPERPARAMETERS ---
INTENSITY_RATIO_THRESHOLD = 1.5      # Overlap override trigger
PROMPT_REPEAT_WINDOW_SEC = 3.5       # Slidng time window for prompt-repeat
PROMPT_REPEAT_LEXICAL_LIMIT = 0.7   # 70% overlap triggers lexical match

# Armenian Particle Lookup Table for Hard-coded Anchoring
PARTICLE_LOOKUP = {
    "ու": "merge_next",            # and / then -> join following segment
    "հա՞": "merge_prev",           # really? / right? -> join previous speaker
    "էլ": "merge_next",            # also / too -> join following segment
    "դե": "clamp_boundary",        # well / so -> tightly wrap timestamps
    "չէ՞": "merge_prev",           # isn't it? -> join previous speaker
    "բան": "clamp_boundary"        # stuff / y'know -> stay local but clamp
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing V77 Pipeline on Device: {device}")

# --- STEP 1: INITIAL COMPONENT LOADING ---
try:
    # Load WhisperX and alignment models
    model = whisperx.load_model("large-v3", device, compute_type="float16")
    waveform, sample_rate = torchaudio.load(AUDIO_PATH)
    audio_data = whisperx.load_audio(AUDIO_PATH)
    
    model_a, metadata = whisperx.load_align_model(
        model_name="Wav2Vec2-Large-LV-60", language=LANGUAGE, device=device
    )
    
    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        text_to_align = f.read()
    
    print("Running WhisperX Forced Alignment...")
    result_aligned = whisperx.align([{'text': text_to_align}], model_a, audio_data, device)
    alignment_segments = result_aligned.get('segments', [])
    
    # Load Pyannote Diarization (v2.1 API compatible with pyannote.audio 3.3.2)
    print("Running Pyannote Speaker Diarization...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
    diarization_pipeline.to(torch.device(device))
    diarization_result = diarization_pipeline(AUDIO_PATH)
    
except Exception as e:
    print(f"!!! Initialization Pipeline Failed: {e}")
    sys.exit(1)

# --- STEP 2: HELPER FUNCTIONS FOR V77 LOGIC LAYER ---

def get_audio_energy(start_sec, end_sec):
    """
    Core Formula: E(s,t) = mean(audio_frame^2)
    Calculates the Root-Mean-Square energy of a specific time frame.
    """
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)
    if start_frame >= end_frame or start_frame >= waveform.shape[1]:
        return 0.0
    
    # Extract frame slice from channel 0
    frame_slice = waveform[0, start_frame:end_frame].numpy()
    if len(frame_slice) == 0:
        return 0.0
    return float(np.mean(frame_slice ** 2))

def get_lexical_overlap(text1, text2):
    """
    Calculates token-level Jaccard similarity for Prompt-Repeat patterns.
    """
    words1 = set(text1.strip().lower().split())
    words2 = set(text2.strip().lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

# --- STEP 3: THE V77 POST-PROCESSING ENGINE ---
print("Executing V77 Heuristic Merging Engine...")

# Build standard pyannote annotation timeline
diarization_annotation = Annotation(uri="audio_file", modality="speaker")
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    diarization_annotation[turn] = speaker

processed_segments = []
history_window = [] # Stores recent segments for Prompt-Repeat matching

for idx, segment in enumerate(alignment_segments):
    seg_start = segment['start']
    seg_end = segment['end']
    text = segment['text'].strip()
    
    if not text:
        continue
        
    current_segment = Segment(seg_start, seg_end)
    overlap = diarization_annotation.crop(current_segment)
    
    # 1. Overlap Resolution via Intensity-Ratio Trigger
    assigned_speaker = 'UNKNOWN'
    if overlap:
        speakers_in_window = list(set(overlap.labels()))
        
        if len(speakers_in_window) > 1:
            # Multi-speaker collision: Evaluate E(s,t) = mean(audio_frame^2)
            energies = {spk: get_audio_energy(seg_start, seg_end) for spk in speakers_in_window}
            sorted_spks = sorted(energies.items(), key=lambda x: x[1], reverse=True)
            
            top_spk, top_energy = sorted_spks[0]
            runner_spk, runner_energy = sorted_spks[1]
            
            # Check ratio threshold (1.5)
            if runner_energy > 0 and (top_energy / runner_energy) > INTENSITY_RATIO_THRESHOLD:
                assigned_speaker = top_spk # Dominated speaker overrides
            else:
                # Default fallback: Max duration winner
                max_dur = 0
                for s, _, spk in overlap.itertracks(yield_label=True):
                    if s.duration > max_dur:
                        max_dur = s.duration
                        assigned_speaker = spk
        else:
            # Single speaker track
            assigned_speaker = speakers_in_window[0]
            
    # 2. Hard-coded Particle Anchoring
    if text in PARTICLE_LOOKUP:
        strategy = PARTICLE_LOOKUP[text]
        if strategy == "merge_prev" and processed_segments:
            assigned_speaker = processed_segments[-1]['speaker']
        elif strategy == "merge_next" and idx + 1 < len(alignment_segments):
            # Temporarily look ahead (simplified fallback to match flow)
            next_overlap = diarization_annotation.crop(Segment(alignment_segments[idx+1]['start'], alignment_segments[idx+1]['end']))
            if next_overlap:
                assigned_speaker = list(set(next_overlap.labels()))[0]
        elif strategy == "clamp_boundary":
            # Shrink tracking margins slightly to shield from background drift
            seg_start += 0.05
            seg_end -= 0.05

    # 3. Prompt-Repeat Sequential Window Comparison
    # Prune sliding history window to maintain the 3.5s limit
    history_window = [h for h in history_window if (seg_start - h['end']) <= PROMPT_REPEAT_WINDOW_SEC]
    
    for hist_seg in history_window:
        if assigned_speaker != hist_seg['speaker']: # Only look for speaker switches
            lex_overlap = get_lexical_overlap(text, hist_seg['text'])
            if lex_overlap >= PROMPT_REPEAT_LEXICAL_LIMIT:
                # Prompt-repeat identified! Compress timeline boundaries to anchor the response
                seg_start = hist_seg['end'] + 0.01 
                break

    # Format speaker ID (SPEAKER_00 -> Speaker 1)
    if assigned_speaker != 'UNKNOWN':
        spk_num = int(assigned_speaker.split('_')[-1]) + 1
        speaker_label = f"Speaker {spk_num}"
    else:
        speaker_label = 'UNKNOWN'

    current_output = {
        'start': seg_start,
        'end': seg_end,
        'speaker': speaker_label,
        'text': text
    }
    
    processed_segments.append(current_output)
    history_window.append(current_output)

# --- STEP 4: OUTPUT EXPORT ---
def format_time(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds / 60) % 60
    h = int(seconds / 3600)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

final_output_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_V77_FINAL_GOLD.txt")
with open(final_output_path, 'w', encoding='utf-8') as f:
    for item in processed_segments:
        line = f"[{format_time(item['start'])} --> {format_time(item['end'])}] ({item['speaker']}) {item['text']}\n"
        f.write(line)

print(f"V77 Gold Standard Output successfully compiled at: {final_output_path}")
