import os, re, torch, time
from pyannote.audio import Pipeline
from pyannote.core import Segment
from collections import Counter

# --- Configuration ---
AUDIO_PATH = "./data/02tarm011225.mp3" 
TXT_PATH = "./data/02 tarm 01 12 25 final.txt"
OUTPUT_DIR = "./output"
SILENCE_GAP = 0.5 # Increased tolerance as requested by Leonora

os.makedirs(OUTPUT_DIR, exist_ok=True)

def time_to_sec(t_str):
    t_str = t_str.replace(',', '.')
    h, m, s = t_str.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def format_srt_time(sec):
    ms = int((sec - int(sec)) * 1000)
    return f"{time.strftime('%H:%M:%S', time.gmtime(sec))},{ms:03d}"

# --- Step 1: Load Transcripts ---
all_words_raw = []
with open(TXT_PATH, 'r', encoding='utf-8') as f:
    content = f.read()
pattern = re.compile(r"(\d+)\n(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.+)")
for m in pattern.findall(content):
    all_words_raw.append({'word': m[3].strip(), 'start': time_to_sec(m[1]), 'end': time_to_sec(m[2])})

# --- Step 2: Speaker Diarization ---
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
diarization = pipeline(AUDIO_PATH) 

spk_counts = Counter()
for label in diarization.labels():
    spk_counts[label] = diarization.label_duration(label)
top_3_raw = [label for label, dur in spk_counts.most_common(3)]

# Mapping logic: 1=Intro, 2=Main Speaker (Mic-2), 3=Commentator (Mic-3)
raw_spk1 = top_3_raw[0] 
raw_spk2 = top_3_raw[1] 
raw_spk3 = top_3_raw[2] if len(top_3_raw) > 2 else None

# --- Step 3: Alignment and Local Refinement (Optimized for noise and misidentification) ---
final_assigned = []
for i, w in enumerate(all_words_raw):
    win = Segment(w['start'], w['end'])
    res = diarization.crop(win)
    clean_word = w['word'].strip(',.?! ՞ :')
    
    # Base AI classification
    if res:
        best_label = res.argmax()
        if best_label == raw_spk1: assigned_spk = 2 
        elif best_label == raw_spk2: assigned_spk = 3
        else: assigned_spk = 1
    else:
        assigned_spk = 2

    # Fix 1: 50s-52s Overlap zone (Ensure semantic alignment)
    if 50.1 <= w['start'] <= 52.1:
        if clean_word in ["Ահա", "հիմիմ", "ասածս", "Ասածս"]:
            assigned_spk = 2 
        else:
            assigned_spk = 3 

    # Fix 2: 05:37 Speaker 1 correction (Identify as backchanneling)
    elif 337.0 <= w['start'] <= 338.0 and clean_word == "այսպես":
        assigned_spk = 3

    # Fix 3: 05:53 Correction to Speaker 3
    elif 353.0 <= w['start'] <= 354.5 and clean_word in ["ի", "վիճակի", "է"]:
        assigned_spk = 3

    # Fix 4: 06:04 Noise filtering (Resolve "ընդեղ" misidentification)
    # Merged into Speaker 2 as this is noise/filler and Speaker 1 is silent
    elif 364.5 <= w['start'] <= 365.2 and clean_word == "ընդեղ":
        assigned_spk = 2

    # Logic A: Intro Protection (Mic-1 recorded introduction)
    if w['start'] < 12.0: 
        assigned_spk = 1

    final_assigned.append({'word': w['word'], 'start': w['start'], 'end': w['end'], 'spk': assigned_spk})

# --- Step 4: Merge and Output ---
output_path = f"{OUTPUT_DIR}/02tarm011225_V85_FINAL.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    if final_assigned:
        curr_spk, curr_start, curr_text = final_assigned[0]['spk'], final_assigned[0]['start'], [final_assigned[0]['word']]
        for i in range(1, len(final_assigned)):
            w = final_assigned[i]
            # Split only if speaker changes or silence exceeds SILENCE_GAP
            if (w['spk'] != curr_spk) or (w['start'] - final_assigned[i-1]['end'] >= SILENCE_GAP):
                f.write(f"{format_srt_time(curr_start)} --> {format_srt_time(final_assigned[i-1]['end'])} [Speaker {curr_spk}]\n{' '.join(curr_text)}\n\n")
                curr_spk, curr_start, curr_text = w['spk'], w['start'], [w['word']]
            else:
                curr_text.append(w['word'])
        f.write(f"{format_srt_time(curr_start)} --> {format_srt_time(final_assigned[-1]['end'])} [Speaker {curr_spk}]\n{' '.join(curr_text)}\n\n")

print(f"Final Version generated")
