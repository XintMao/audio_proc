import sys
import torch
import whisperx
from pyannote.audio import Pipeline
import json
import os
from pyannote.core import Segment, Annotation
import torchaudio # 保持导入 torchaudio，即使在 Colab 环境中它可能不是必需的，但有助于依赖解析

# --- 必须导入所有用于白名单的模块 (解决 PyTorch 2.6+ 的安全加载问题) ---
import torch.serialization 
import omegaconf.listconfig 
import omegaconf.base
import omegaconf.nodes
import typing
import collections
# 注意：Colab 环境可能不需要 pytorch_lightning 相关的白名单，但为了安全起见，我们添加常用的 omegaconf 依赖
import omegaconf.dictconfig

# --- 启用核心白名单 (解决 Pyannote 模型加载的 WeightsUnpickler error) ---
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        # omegaconf 依赖 (核心报错源)
        omegaconf.listconfig.ListConfig, 
        omegaconf.base.ContainerMetadata,
        omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata,
        omegaconf.dictconfig.DictConfig,
        # 其他 Python 内置类型 (通常安全)
        list,
        dict,
        int,
        typing.Any,
        collections.defaultdict,
    ])
    print("已启用核心白名单以解决 Pyannote 模型的安全加载问题。")

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

# -------------------------------------------------------------
# --- 1. 运行 WhisperX 强制对齐 (Forced Alignment) ---
# -------------------------------------------------------------
print("\n--- 1. 运行 WhisperX 强制对齐 (Forced Alignment) ---")

# 定义分段结果文件的路径，便于后续步骤引用
alignment_json_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_alignment.json")
resegmented_json_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_resegmented_alignment.json")
rttm_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_diarization.rttm")


try:
    # 1. 加载 Whisper 模型
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(AUDIO_PATH) 

    # 2. 定义 Wav2Vec2.0 对齐模型的 ID
    ARMENIAN_ALIGN_MODEL_ID = "facebook/wav2vec2-base"
    
    # 3. 加载 Wav2Vec2.0 模型进行对齐
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

    # 保存原始对齐结果 (包含词级别时间戳)
    with open(alignment_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=4)
    print(f"原始对齐结果已保存到: {alignment_json_path}")


    # -------------------------------------------------------------
    # --- 1.5. 导师要求：基于长度和停顿的重新分段 ---
    # -------------------------------------------------------------
    
    # 核心分段逻辑，放在 merge_and_format 函数上方
    # (此部分逻辑放在后面定义)

    print("\n--- 1.5. 按最大长度重新分段 (最大30秒，停顿切割) ---")
    
    # 载入刚才保存的完整对齐结果
    with open(alignment_json_path, 'r', encoding='utf-8') as f:
        full_aligned_result = json.load(f)

    # 执行重新分段
    resegmented_result = split_segments_by_length(full_aligned_result)
    
    # 将重新分段的结果保存到一个新的 JSON 文件
    with open(resegmented_json_path, 'w', encoding='utf-8') as f:
        json.dump(resegmented_result, f, ensure_ascii=False, indent=4)
    print(f"重新分段结果已保存到: {resegmented_json_path}")
    
except Exception as e:
    print(f"!!! WhisperX 运行失败。")
    print(f"详细错误: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# --- 2. 运行 pyannote 说话人识别 (Speaker Diarization) ---
# -------------------------------------------------------------
print("\n--- 2. 运行 pyannote 说话人识别 ---")

try:
    # Colab 环境使用最新版本的 pyannote/speaker-diarization
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization") 
    diarization_pipeline.to(torch.device(device)) 

    diarization_result = diarization_pipeline(AUDIO_PATH)

    # 保存说话人识别结果 RTTM
    with open(rttm_path, "w") as f:
        diarization_result.write_rttm(f)
    print(f"说话人识别结果已保存到: {rttm_path}")
    
except Exception as e:
    print(f"!!! Pyannote Diarization 运行失败。请务必确认已完成 `huggingface-cli login`。")
    print(f"详细错误: {e}")
    sys.exit(1)


# -------------------------------------------------------------
# --- 3. 核心分段与格式化函数定义 ---
# -------------------------------------------------------------

MAX_SEGMENT_DURATION = 30.0 # 导师要求的最大时长 (秒)
MIN_PAUSE_FOR_SPLIT = 0.5    # 定义最小停顿时间作为自然的语音停顿点 (秒)

def split_segments_by_length(aligned_data, max_duration=MAX_SEGMENT_DURATION, min_pause=MIN_PAUSE_FOR_SPLIT):
    """
    根据最大时长和自然语音停顿，将 WhisperX 的对齐结果重新分段。
    目标：确保每个片段不超过 max_duration，并在词语之间的停顿处切割。
    """
    new_segments = []
    
    # 将所有词语展平为一个列表，方便遍历
    all_words = []
    for segment in aligned_data.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    if not all_words:
        return {'segments': []}

    current_start_time = all_words[0]['start']
    current_segment_words = []
    
    for i, word in enumerate(all_words):
        
        current_segment_words.append(word)
        current_duration = word['end'] - current_start_time
        
        is_long_pause = False
        if i < len(all_words) - 1:
            next_word_start = all_words[i+1]['start']
            pause_duration = next_word_start - word['end']
            if pause_duration > min_pause:
                is_long_pause = True
        
        is_end_of_words = (i == len(all_words) - 1)
        
        # 满足切割条件：时长达到限制 AND 存在长停顿 OR 已经是最后一个词
        if (current_duration >= max_duration and is_long_pause) or is_end_of_words:
            
            # 确定新片段的结束时间 (当前片段最后一个词的结束时间)
            new_end_time = current_segment_words[-1]['end']
            
            # 构造新的片段文本
            new_text = " ".join([w['word'] for w in current_segment_words])

            new_segments.append({
                'start': current_start_time,
                'end': new_end_time,
                'text': new_text,
                'words': current_segment_words # 保留 words 列表便于调试
            })
            
            # 重置，为下一个片段做准备
            if not is_end_of_words:
                current_start_time = all_words[i+1]['start']
                current_segment_words = []
            else:
                break 

    return {'segments': new_segments}


def format_time_srt(seconds):
    """将秒转换为 HH:MM:SS,mmm 格式"""
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = int(seconds / 60) % 60
    h = int(seconds / 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" 

def merge_and_format(alignment_path, rttm_path):
    """核心逻辑：将带时间戳的片段与说话人分段进行时间重叠匹配。"""
    
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

# -------------------------------------------------------------
# --- 4. 执行合并和保存最终结果 ---
# -------------------------------------------------------------

try:
    # 核心修改：使用重新分段后的 JSON 文件作为输入
    combined_results = merge_and_format(resegmented_json_path, rttm_path) 

    final_output_path = os.path.join(OUTPUT_DIR, f"{BASE_FILENAME}_final_transcript_formatted.txt")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        f.writelines(combined_results)

    print(f"\n✅ 任务完成！最终匹配导师格式的转录文本已保存到: {final_output_path}")

except Exception as e:
    print(f"!!! 最终结果合并失败。")
    print(f"详细错误: {e}")
    sys.exit(1)