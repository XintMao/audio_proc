import torch
import whisperx
from pyannote.audio import Pipeline
import json
import os
import sys
from pyannote.core import Segment, Annotation

# --- 配置参数 ---
BASE_FILENAME = "01281_ARM" 
AUDIO_PATH = f"./data/{BASE_FILENAME}.mp3" 
TRANSCRIPT_PATH = f"./data/{BASE_FILENAME}.txt"
OUTPUT_DIR = "./output"
LANGUAGE = "hy" # 亚美尼亚语

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- 配置信息 ---")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"目标语言: {LANGUAGE} (亚美尼亚语)")

# GPU 配置 (Colab T4 推荐配置)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 
compute_type = "float16" if torch.cuda.is_available() else "float32"

# --- 1. 运行 WhisperX 强制对齐 (Forced Alignment) ---
print("\n--- 1. 运行 WhisperX 强制对齐 (Forced Alignment) ---")

try:
    # 1. 加载 Whisper 模型
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(AUDIO_PATH) 

    # 2. 定义 Wav2Vec2.0 对齐模型的 ID
    ARMENIAN_ALIGN_MODEL_ID = "facebook/wav2vec2-base"
    
    # 3. 加载 Wav2Vec2.0 模型进行对齐
    # Colab 环境通常不会有兼容性问题，所以直接加载即可
    model_a, metadata = whisperx.load_align_model(
        language_code=LANGUAGE, 
        model_name=ARMENIAN_ALIGN_MODEL_ID, 
        device=device
    )

    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        text_to_align = f.read()

    align_input = [{'text': text_to_align}]

    # 执行强制对齐：参数简洁，不会报错
    result_aligned = whisperx.align(align_input, model_a, audio, device) 

    alignment_json_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_alignment.json")
    with open(alignment_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=4)
    print(f"对齐结果已保存到: {alignment_json_path}")

except Exception as e:
    print(f"!!! WhisperX 运行失败。")
    print(f"详细错误: {e}")
    sys.exit(1)


# --- 2. 运行 pyannote 说话人识别 (Speaker Diarization) ---
print("\n--- 2. 运行 pyannote 说话人识别 ---")

try:
    # Colab 环境使用最新版本的 pyannote/speaker-diarization
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization") 
    diarization_pipeline.to(torch.device(device)) 

    diarization_result = diarization_pipeline(AUDIO_PATH)

    rttm_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_diarization.rttm")
    with open(rttm_path, "w") as f:
        diarization_result.write_rttm(f)
    print(f"说话人识别结果已保存到: {rttm_path}")
    
except Exception as e:
    print(f"!!! Pyannote Diarization 运行失败。请务必确认已完成 `huggingface-cli login`。")
    print(f"详细错误: {e}")
    sys.exit(1)

# --- 3. 结果合并和格式化 (匹配导师要求格式) ---

def format_time_srt(seconds):
    """将秒转换为 HH:MM:SS,mmm 格式"""
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds / 60) % 60
    h = int(seconds / 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" 

def merge_and_format(alignment_path, rttm_path):
    """核心逻辑：将带时间戳的文本片段与说话人分段进行时间重叠匹配。"""
    
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
                speaker_id_num = int(speaker.split('_')[-1]) + 1 
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

# --- 4. 执行合并和保存最终结果 ---

try:
    combined_results = merge_and_format(alignment_json_path, rttm_path)

    final_output_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_final_transcript_formatted.txt")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        f.writelines(combined_results)

    print(f"\n✅ 任务完成！最终匹配导师格式的转录文本已保存到: {final_output_path}")

except Exception as e:
    print(f"!!! 最终结果合并失败。")
    print(f"详细错误: {e}")
    sys.exit(1)