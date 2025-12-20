import os
import re
import torch
import json
from pyannote.audio import Pipeline
from pyannote.core import Segment
import time

# --- 关键配置 ---
BASE_FILENAME = "01_kapital_29_09_25"
AUDIO_PATH = f"./data/{BASE_FILENAME}.mp3"
TXT_PATH = f"./data/{BASE_FILENAME}.txt"  # 亚美尼亚团队提供的词级时间戳文件
OUTPUT_DIR = "./output"
MAX_DURATION = 30.0    # 导师要求的 30s 逻辑
SILENCE_GAP = 0.5      # 导师要求的 VAD/停顿切分阈值

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def time_to_sec(t_str):
    t_str = t_str.replace(',', '.')
    h, m, s = t_str.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def parse_armenian_txt(path):
    """解析词级时间戳 TXT [cite: 72-103]"""
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(r"(\d+)\n(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.+)")
    for m in pattern.findall(content):
        words.append({'word': m[3].strip(), 'start': time_to_sec(m[1]), 'end': time_to_sec(m[2])})
    return words

def format_srt_time(sec):
    ms = int((sec - int(sec)) * 1000)
    return f"{time.strftime('%H:%M:%S', time.gmtime(sec))},{ms:03d}"

# 1. 加载数据
print("--- Step 1: Loading Data ---")
all_words = parse_armenian_txt(TXT_PATH)

# 2. 运行 Diarization (Step 3)
print("--- Step 2: Running Pyannote Diarization ---")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
pipeline.to(torch.device(device))
diarization = pipeline(AUDIO_PATH)

# 3. 整合逻辑 (Step 4)
print("--- Step 3: Merging & VAD Splitting ---")
final_segments = []
current_words = []
start_t = all_words[0]['start']

for i, w in enumerate(all_words):
    current_words.append(w)
    # VAD 逻辑：检查词间停顿或达到 30s
    is_last = i == len(all_words) - 1
    has_gap = not is_last and (all_words[i+1]['start'] - w['end'] >= SILENCE_GAP)
    is_long = (w['end'] - start_t) >= MAX_DURATION

    if has_gap or is_long or is_last:
        # 确定说话人 (argmax overlap)
        win = Segment(start_t, w['end'])
        speakers = diarization.crop(win)
        spk_label = f"[Speaker {int(speakers.argmax().split('_')[-1])+1}]" if speakers else "[UNKNOWN]"
        
        text = " ".join([word['word'] for word in current_words])
        final_segments.append(f"{format_srt_time(start_t)} --> {format_srt_time(w['end'])} {spk_label}\n{text}\n\n")
        
        if not is_last:
            current_words = []
            start_t = all_words[i+1]['start']

# 4. 保存
with open(f"{OUTPUT_DIR}/{BASE_FILENAME}_final.txt", 'w', encoding='utf-8') as f:
    f.writelines(final_segments)
print(f"✅ Success! Output: {OUTPUT_DIR}/{BASE_FILENAME}_final.txt")
