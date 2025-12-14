import sys
import torch
import whisperx
from pyannote.audio import Pipeline
import json
import os
from pyannote.core import Segment, Annotation
import torchaudio 

# --- 必须导入所有用于白名单的模块 (解决 PyTorch 2.6+ 的安全加载问题) ---
import torch.serialization 
import omegaconf.listconfig 
import omegaconf.base
import omegaconf.nodes
import typing
import collections
import omegaconf.dictconfig
import torch.torch_version # 核心修复

# --- 启用核心白名单 (解决 Pyannote 模型加载的 WeightsUnpickler error) ---
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        # omegaconf 依赖
        omegaconf.listconfig.ListConfig, 
        omegaconf.base.ContainerMetadata,
        omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata,
        omegaconf.dictconfig.DictConfig,
        
        # 新增项：解决 torch.torch_version 报错
        torch.torch_version.TorchVersion, 
        
        # 其他 Python 内置类型 
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
MAX_SEGMENT_DURATION = 30.0 # 导师要求的最大时长 (秒)

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
# --- 3. 核心分段与格式化函数定义 ---
# -------------------------------------------------------------

def split_segments_by_length(aligned_data, max_duration=MAX_SEGMENT_DURATION):
    """
    根据导师提供的逻辑，将 WhisperX 的对齐结果重新分段。
    目标：确保每个片段不超过 max_duration，并在单词边界处切割。
    """
    
    # 1. 将所有词语展平为一个列表
    all_words = []
    for segment in aligned_data.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    if not all_words:
        return {'segments': []}

    # 2. 生成 (start, end) 边界列表 (导师的核心逻辑)
    chunks = []
    current_chunk_start = all_words[0]['start']
    prev_word = None

    for word in all_words:
        word_end = word["end"]

        # 如果加入这个词会超过最大长度限制
        if word_end - current_chunk_start > max_duration:
            # 确保这不是第一个词（防止prev_word是None）
            if prev_word is not None:
                # 在前一个词的结束时间处关闭当前块
                chunks.append((current_chunk_start, prev_word["end"]))
                # 新的块从前一个词的结束时间开始
                current_chunk_start = prev_word["end"] 
        
        prev_word = word

    # 添加最后一个块
    if prev_word is not None:
        chunks.append((current_chunk_start, prev_word["end"]))
    
    # 3. 根据边界列表重建 JSON 片段结构
    new_segments = []
    word_index = 0
    
    for start_time, end_time in chunks:
        segment_words = []
        
        # 从当前位置开始，找到所有落在新边界内的词
        while word_index < len(all_words) and all_words[word_index]['end'] <= end_time:
            segment_words.append(all_words[word_index])
            word_index += 1
            
        if segment_words:
            # 构造新的片段文本
            new_text = " ".join([w['word'] for w in segment_words])

            new_segments.append({
                'start': start_time,
                'end': end_time,
                'text': new_text,
                'words': segment_words # 保留 words 列表
            })
            
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
                # pyannote 从 0 开始编号，我们显示从 1 开始
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

    # 执行强制对齐
    result_aligned = whisperx.align(align_input, model_a, metadata, audio, device) 

    # 保存原始对齐结果 (包含词级别时间戳)
    with open(alignment_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_aligned, f, ensure_ascii=False, indent=4)
    print(f"原始对齐结果已保存到: {alignment_json_path}")


    # -------------------------------------------------------------
    # --- 1.5. 导师要求：基于长度的重新分段 ---
    # -------------------------------------------------------------
    
    print(f"\n--- 1.5. 按最大长度重新分段 (最大{MAX_SEGMENT_DURATION}秒) ---")
    
    # 载入刚才保存的完整对齐结果
    with open(alignment_json_path, 'r', encoding='utf-8') as f:
        full_aligned_result = json.load(f)

    # 执行重新分段 (使用导师提供的逻辑)
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
