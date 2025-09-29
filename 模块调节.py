

import os
import warnings
from typing import List, Optional
from pydub import AudioSegment
import shutil
from gradio_client import Client, file
import joblib
import json
import subprocess
from sklearn.linear_model import LinearRegression
import numpy as np
import soundfile as sf
import pyrubberband as rb

# --- 配置参数 ---
TRANSFORMERS_LINE = 2
TIME_LINE = 1
ENGLISH_LINE = -1
SRT_PATH = r"D:\aliyun\ict2023"
REF_AUDIO_PATH = os.path.join(SRT_PATH, "REF_AUDIO_PATH")
OUTPUT_PATH = SRT_PATH
TMP_DIR = os.path.join(SRT_PATH, "tmp")

MAX_SUBTITLE_LENGTH = 49  # 设置单个字幕的最大字符长度

# === 新增：当前任务状态文件 ===
STATUS_FILE = os.path.join(TMP_DIR, "dubbing_status.json") # 放在总的tmp目录下

# === 机器学习模型配置 ===
MODEL_PATH = os.path.join(SRT_PATH, "duration_predictor.joblib")
TRAINING_THRESHOLD = 5




# === 速度安全范围 ===
MIN_SPEED, MAX_SPEED = 0.7, 1.5






def load_status(status_file: str) -> dict:
    """加载当前状态文件。如果不存在或无效，返回空字典。"""
    if not os.path.exists(status_file):
        return {}
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_status(status_file: str, file_name: str, index: int):
    """保存当前任务的进度。这会覆盖之前的所有内容。"""
    try:
        # 确保父目录存在
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"current_file": file_name, "last_completed_index": index}, f, indent=4)
    except IOError as e:
        print(f"!! 严重错误：无法保存状态到 '{status_file}'。错误: {e}")

def clear_status(status_file: str):
    """任务成功完成后，清除状态文件。"""
    if os.path.exists(status_file):
        try:
            os.remove(status_file)
        except OSError as e:
            print(f"!! 警告：无法删除状态文件 '{status_file}'。错误: {e}")


# ... existing code ...
def check_ffmpeg():
    """检查 ffmpeg 是否安装并可用"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True, encoding='utf-8')
        print("ffmpeg 已找到。")
        return True
    except FileNotFoundError:
        print("错误：ffmpeg 未找到。请确保它已安装并添加到系统 PATH 中。")
        return False
    except subprocess.CalledProcessError as e:
        if "ffmpeg version" in e.stderr.lower() or "ffmpeg version" in e.stdout.lower():
            print("ffmpeg 已找到 (执行 version 命令时可能输出了警告/错误，但可执行)。")
            return True
        print(f"错误：ffmpeg 执行时出错 (但可能已安装)。尝试检查 ffmpeg -version 手动。\n{e.stderr}")
        return False
    except Exception as e_gen:
        print(f"检查 ffmpeg 时发生未知错误: {e_gen}")
        return False


def merge_audio_video(input_folder, output_folder):
    """
    在指定输入文件夹的各层子文件夹中查找位于同一目录下的同名 .wav 和视频文件
    (如 .mp4, .ts, .mkv)，将 .wav 作为音频合并到视频中，并将结果以 .mp4 格式
    保存到输出文件夹，同时保留原始的子目录结构。

    参数:
    input_folder (str): 包含源媒体文件的文件夹路径。
    output_folder (str): 保存合并后文件的文件夹路径 (将在此路径下创建原始子目录结构)。
    """
    if not check_ffmpeg():
        return

    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
        return

    # 定义支持的视频文件扩展名 (可以按需添加)
    VIDEO_EXTENSIONS = {".mp4", ".ts", ".mkv", ".mov", ".avi"}
    print(f"将查找以下视频文件类型: {', '.join(VIDEO_EXTENSIONS)}")

    print(f"输入文件夹：'{input_folder}'")
    print(f"输出文件夹根目录：'{output_folder}'")

    merged_count = 0
    skipped_count = 0

    print(f"\n正在扫描文件夹 '{input_folder}' 并处理...")
    for root, _, files in os.walk(input_folder):
        # media_files 存储在当前 'root' 目录下找到的 wav 和视频文件
        # 结构: { "basename": {"wav": "path/to/file.wav", "video": "path/to/video.ts"} }
        media_files_in_current_dir = {}

        for filename in files:
            base, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext == ".wav" or ext in VIDEO_EXTENSIONS:
                full_path = os.path.join(root, filename)
                if base not in media_files_in_current_dir:
                    media_files_in_current_dir[base] = {}

                if ext == ".wav":
                    media_files_in_current_dir[base]['wav'] = full_path
                elif ext in VIDEO_EXTENSIONS:
                    # 如果已找到一个同名的视频文件，发出警告并使用第一个找到的
                    if 'video' in media_files_in_current_dir[base]:
                        print(f"  警告: 在目录 '{root}' 中为 '{base}' 找到了多个视频文件。")
                        print(f"    已记录: {os.path.basename(media_files_in_current_dir[base]['video'])}")
                        print(f"    跳过:   {filename}")
                    else:
                        media_files_in_current_dir[base]['video'] = full_path

        # 处理在当前目录 'root' 中找到的匹配对
        for base_name, paths in media_files_in_current_dir.items():
            if 'wav' in paths and 'video' in paths:
                wav_file = paths['wav']
                video_file = paths['video']

                # 确定输出子目录结构
                relative_dir_path = os.path.relpath(root, input_folder)
                output_sub_dir = os.path.join(output_folder, relative_dir_path)
                os.makedirs(output_sub_dir, exist_ok=True)

                # 统一输出为 .mp4 格式
                output_filename = base_name + ".mp4"
                output_file_path = os.path.join(output_sub_dir, output_filename)

                print(f"\n在目录 '{root}' 中找到匹配对 '{base_name}':")
                print(f"  视频源: {video_file}")
                print(f"  音频源: {wav_file}")
                print(f"  输出到: {output_file_path}")

                cmd = [
                    "ffmpeg",
                    "-i", video_file,
                    "-i", wav_file,
                    # 直接复制视频流，不进行重新编码，速度极快。
                    # 如果遇到错误，特别是从 .ts 或 .avi 等格式转换时，
                    # 可能是视频编解码器与 .mp4 容器不兼容。
                    # 此时可尝试将其改为重新编码，例如: "-c:v", "libx264"
                    "-c:v", "copy",
                    # 将 WAV 音频编码为 AAC，这是 MP4 的标准音频格式
                    "-c:a", "aac",
                    "-map", "0:v:0",  # 从第一个输入 (视频文件) 选择视频流
                    "-map", "1:a:0",  # 从第二个输入 (wav 文件) 选择音频流
                    "-shortest",      # 以最短的输入流长度为准，防止时长不匹配
                    "-y",             # 覆盖已存在的输出文件
                    output_file_path
                ]

                try:
                    process = subprocess.run(cmd, capture_output=True, check=True, text=True, encoding='utf-8')
                    print(f"  成功合并 '{base_name}' 到 '{output_file_path}'.")

                    if process.stderr and process.stderr.strip():
                        stderr_lines = process.stderr.strip().split('\n')
                        if len(stderr_lines) > 8 and not any(kw in process.stderr.lower() for kw in ['error', 'failed']):
                            print(f"  ffmpeg 完成，在 stderr 上有详细输出 (共 {len(stderr_lines)} 行)。最后几行为:")
                            indented_summary = "\n    ".join(stderr_lines[-3:])
                            print(f"    {indented_summary}")
                        else:
                            indented_stderr = "\n    ".join(stderr_lines)
                            print(f"  ffmpeg 信息 (来自 stderr):\n    {indented_stderr}")
                    merged_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"  错误：合并 '{base_name}' 失败。")
                    print(f"  ffmpeg 命令: {' '.join(cmd)}")
                    print(f"  ffmpeg 返回码: {e.returncode}")
                    if e.stdout and e.stdout.strip():
                        print(f"  ffmpeg 标准输出:\n   ")
                    if e.stderr and e.stderr.strip():
                        print(f"  ffmpeg 标准错误:\n   ")
                    skipped_count += 1
                except Exception as e_gen:
                    print(f"  处理 '{base_name}' 时发生意外本地错误: {e_gen}")
                    skipped_count += 1

    print(f"\n处理完成。")
    print(f"成功合并 {merged_count} 个文件对。")
    if skipped_count > 0:
        print(f"跳过或失败 {skipped_count} 个文件对。")

    if merged_count == 0 and skipped_count == 0:
        has_walked_files = False
        for _, _, files_in_walk in os.walk(input_folder):
            if files_in_walk:
                has_walked_files = True
                break
        if not has_walked_files:
            print("输入文件夹中没有找到任何文件。")
        else:
            print("在输入文件夹中找到了文件，但未找到任何在相同目录下同名的 WAV 和视频文件对。")
def extract_audio_from_directory(video_dir, audio_dir, output_format="wav"):
    """
    遍历指定目录下的所有视频文件，提取音频并保存到另一个目录。

    :param video_dir: 包含视频文件的目录路径。
    :param audio_dir: 保存提取出的音频文件的目录路径。
    :param output_format: 输出音频的格式，这里我们主要使用 'wav'。
    """
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(audio_dir, exist_ok=True)
    print(f"音频将保存到: {audio_dir}")

    # 支持的视频文件扩展名列表
    supported_video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')

    # 遍历视频目录下的所有文件
    for filename in os.listdir(video_dir):
        # 检查文件是否为支持的视频格式
        if filename.lower().endswith(supported_video_extensions):
            video_path = os.path.join(video_dir, filename)

            # 构建输出音频文件的路径
            # 文件名使用原视频名，扩展名替换为指定的音频格式
            audio_filename = f"{os.path.splitext(filename)[0]}.{output_format}"
            audio_path = os.path.join(audio_dir, audio_filename)

            print(f"\n[处理中] ==> {filename}")

            # 如果音频文件已存在，则跳过
            if os.path.exists(audio_path):
                print(f"[已跳过] 音频文件 '{audio_filename}' 已存在。")
                continue

            try:
                # 使用ffmpeg提取音频
                # 构建ffmpeg命令
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', video_path,  # 输入视频文件
                    '-vn',  # 禁用视频流
                    '-acodec', 'pcm_s16le',  # 音频编解码器
                    '-ar', '44100',  # 音频采样率
                    '-ac', '2',  # 音频通道数
                    audio_path,  # 输出音频文件
                    '-y'  # 覆盖输出文件
                ]

                # 执行ffmpeg命令，解决UnicodeDecodeError问题
                # 不使用text=True，而是分别处理stdout和stderr
                result = subprocess.run(ffmpeg_cmd, capture_output=True, encoding='utf-8', errors='ignore')

                # 检查命令执行结果
                if result.returncode == 0:
                    print(f"[成功] ✓ 已提取音频并保存为: {audio_filename}")
                else:
                    # 过滤掉错误输出中的编码问题
                    error_output = result.stderr.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                    print(f"[失败] X 处理文件 '{filename}' 时发生错误: {error_output}")

            except Exception as e:
                print(f"[失败] X 处理文件 '{filename}' 时发生错误: {e}")

    print("\n所有视频处理完成！")


# ==============================================================================
# === 【重大修改】语速预测模型类，实现基于反向计算的精准学习 ===
# ==============================================================================
# ==============================================================================
# === 【重大重构】时长预测模型类，实现更科学的解耦学习 ===
# ==============================================================================
class DurationPredictor:
    def __init__(self, model_path: str, training_threshold: int):
        self.model_path, self.training_threshold = model_path, training_threshold
        self.new_data = []
        self.historic_X, self.historic_y = [], []
        try:
            # 加载新模型，如果失败则创建新模型
            self.model, self.historic_X, self.historic_y = joblib.load(self.model_path)
            print(f"--- 成功加载本地【时长】预测模型。模型已有 {sum(len(x) for x in self.historic_X)} 条历史数据。 ---")
        except (FileNotFoundError, EOFError, ValueError) as e:
            print(f"--- 未找到或无法解析本地时长模型 ({e.__class__.__name__})，将创建一个新模型。 ---")
            self.model = LinearRegression()

    def _get_features(self, text: str) -> np.ndarray:
        """特征提取：现在只依赖于文本本身。"""
        # 为了简单和鲁棒，我们只使用文本长度作为核心特征。
        return np.array([len(text)]).reshape(1, -1)

    def predict_duration(self, text: str) -> float:
        """核心功能：预测文本在1.0倍速下的原始时长（秒）。"""
        # 如果模型还未训练，则给出一个基于字符数的合理估算 (例如, 6个字/秒)
        if not hasattr(self.model, "coef_") or self.model.coef_ is None:
            return len(text) / 6.0
        # 使用模型进行预测
        predicted_duration = self.model.predict(self._get_features(text))[0]
        # 确保预测值是正数
        return max(0.1, predicted_duration)  # 至少返回0.1秒

    def add_data_point_and_retrain(self, text: str, actual_raw_duration_s: float):
        """
        学习步骤：使用【真实的原始时长】作为训练标签。
        """
        if actual_raw_duration_s <= 0: return

        # 训练数据点：{ 特征: 文本长度, 标签: 真实原始时长 }
        self.new_data.append({
            "features": self._get_features(text)[0],
            "duration": actual_raw_duration_s
        })
        print(
            f"  -> 已收集 {len(self.new_data)}/{self.training_threshold} 个新数据点 (真实原始时长: {actual_raw_duration_s:.3f}s)。")

        if len(self.new_data) >= self.training_threshold:
            self.train()

    def train(self):
        if not self.new_data: return
        print(f"--- 达到训练阈值，使用 {len(self.new_data)} 个新数据点更新【时长】模型... ---")
        X_new = np.array([d['features'] for d in self.new_data])
        y_new = np.array([d['duration'] for d in self.new_data])

        # 合并历史数据和新数据
        X_combined = np.concatenate(self.historic_X + [X_new]) if self.historic_X else X_new
        y_combined = np.concatenate(self.historic_y + [y_new]) if self.historic_y else y_new

        self.model.fit(X_combined, y_combined)

        # 更新历史数据记录
        self.historic_X.append(X_new)
        self.historic_y.append(y_new)
        self.new_data = []

        try:
            joblib.dump((self.model, self.historic_X, self.historic_y), self.model_path)
            print(f"--- 时长模型训练完毕并成功保存！总数据点: {len(X_combined)} ---")
        except Exception as e:
            print(f"!! 保存时长模型时发生错误: {e}")


# ==============================================================================
# === 全局初始化与辅助函数 (保持不变) ===
# ==============================================================================
duration_predictor = DurationPredictor(MODEL_PATH, TRAINING_THRESHOLD)
try:
    print("--- 正在连接到 F5-TTS Gradio 服务 (http://127.0.0.1:7860)... ---")
    tts_client = Client("http://127.0.0.1:7860/")
    print("--- 连接成功！ ---")
except Exception as e:
    print(f"!! 致命错误: 无法连接到 Gradio 服务。错误: {e}"); exit()


def time_str_to_seconds(time_str: str) -> float:
    try:
        h, m, s = time_str.split(':'); s, ms = s.split(','); return int(h) * 3600 + int(m) * 60 + int(s) + int(
            ms) / 1000
    except ValueError as e:
        print(f"解析时间错误: {time_str}, 错误: {e}"); return 0.0


def parse_subtitles(file_content: str, transformers_line: int = TRANSFORMERS_LINE) -> List[List]:
    subtitles = file_content.strip().split('\n\n');
    result = []
    for idx, segment in enumerate(subtitles):
        lines = segment.strip().splitlines();
        if len(lines) < TIME_LINE + 1: continue
        time_line = lines[TIME_LINE].strip();
        if " --> " not in time_line: continue
        try:
            start_time, end_time = [t.strip() for t in time_line.split('-->')]
        except Exception as e:
            print(f"段落 {idx} 解析时间错误: {e}"); continue
        if len(lines) <= transformers_line: continue
        text = lines[transformers_line].strip();
        english_text = lines[ENGLISH_LINE].strip()
        result.append([start_time, end_time, text, english_text])
    return result


def crop_audio(start_time_str: str, end_time_str: str, input_file: str) -> Optional[AudioSegment]:
    try:
        start_ms = time_str_to_seconds(start_time_str) * 1000; end_ms = time_str_to_seconds(
            end_time_str) * 1000; audio = AudioSegment.from_file(input_file); return audio[max(0, start_ms):end_ms]
    except Exception as e:
        print(f"!! 裁剪或读取音频时发生错误: {e}"); return None


# 在配置参数部分添加最大字符限制


# ... existing code ...

def merge_consecutive_subtitles(subtitles: List[List]) -> List[List]:
    if not subtitles: return []
    merged = []
    current_start, current_end, current_text, current_english = subtitles[0]

    for i in range(1, len(subtitles)):
        next_start, next_end, next_text, next_english = subtitles[i]

        # 检查是否时间连续且合并后不会超过最大长度限制
        if current_end == next_start and len(current_text + next_text) <= MAX_SUBTITLE_LENGTH:
            current_end = next_end
            current_text += next_text
            current_english += next_english
        else:
            # 如果不满足合并条件，将当前字幕添加到结果中
            merged.append([current_start, current_end, current_text, current_english])
            # 开始新的字幕片段
            current_start, current_end, current_text, current_english = next_start, next_end, next_text, next_english

    # 添加最后一个字幕片段
    merged.append([current_start, current_end, current_text, current_english])
    return merged


def adjust_duration_with_rubberband(input_path: str, output_path: str, target_duration_s: float):
    try:
        y, sr = sf.read(input_path);
        current_duration_s = len(y) / sr
        rate = current_duration_s / target_duration_s
        if abs(rate - 1.0) < 0.001: shutil.copy(input_path, output_path); return
        print(f"    -> 需调节时长：从 {current_duration_s:.2f}s 到 {target_duration_s:.2f}s (速率: {rate:.2f}x)")
        stretched_y = rb.time_stretch(y, sr, rate)
        sf.write(output_path, stretched_y, sr)
        print(f"    -> 时长调节完毕，已保存到: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"!! 时长调节时发生严重错误: {e}");
        shutil.copy(input_path, output_path)


def generate_audio_api(ref_audio_path: str, gen_text: str, speed: float,ref_text:str) -> Optional[str]:
    try:
        result = tts_client.predict(file(ref_audio_path),
                                    # '',
                                    gen_text,
                                    # ref_text,
                                    gen_text, True, True, -1, 0.15, 32, speed,
                                    api_name="/basic_tts")
        path = result[0];
        return path if path and os.path.exists(path) else None
    except Exception as e:
        print(f"\n[ERROR] 调用 Gradio API 时发生错误: {e}"); return None


# === 【重大重构】tts_process_task，采用预测时长->计算速度的新流程 ===
def tts_process_task(index: int, subtitle: List, main_reference_audio: str, current_tmp_dir: str):
    start_time, end_time, text, _ = subtitle
    print(f"\n[Task {index}] 处理字幕: '{text[:30]}...'")

    if not text.strip(): print("  -> 文本为空，跳过。"); return

    reference_clip = crop_audio(start_time, end_time, main_reference_audio)
    if not reference_clip or len(reference_clip) == 0:
        print(f"!! Task {index} 无法提取或参考音频时长为0，跳过。");
        return

    target_duration_s = len(reference_clip) / 1000.0
    print(f"  -> 目标时长: {target_duration_s:.2f}s")

    # --- 新的执行逻辑 ---
    # 步骤1: 预测在1.0x速度下的“原始时长”
    predicted_raw_duration_s = duration_predictor.predict_duration(text)
    print(f"  -> [第1步] 模型预测原始时长: {predicted_raw_duration_s:.2f}s")

    # 步骤2: 根据预测时长和目标时长，计算出理论上需要的速度
    if predicted_raw_duration_s <= 0.1:  # 防止除零错误
        required_speed = 1.0
    else:
        required_speed = predicted_raw_duration_s / target_duration_s

    # 步骤3: 将计算出的速度限制在安全范围内
    applied_speed = max(MIN_SPEED, min(required_speed, MAX_SPEED))
    print(f"  -> [第2步] 计算所需速度: {required_speed:.2f}x -> 应用值: {applied_speed:.2f}x")

    ref_tmp_dir = os.path.join(current_tmp_dir, "ref_clips")
    os.makedirs(ref_tmp_dir, exist_ok=True)
    ref_clip_path = os.path.join(ref_tmp_dir, f"ref_{index}.wav")
    reference_clip.export(ref_clip_path, format="wav")

    # 步骤4: 以应用速度生成音频
    print(f"  -> [第3步] 以 {applied_speed:.2f}x 的速度生成音频...")
    ref_text = _
    # print(ref_text)
    raw_generated_path = generate_audio_api(ref_clip_path, text, applied_speed, ref_text)

    output_path_for_merge = os.path.join(current_tmp_dir, f"output_{index}.wav")

    if raw_generated_path:
        actual_raw_duration_s = 0
        try:
            # 获取生成音频的真实时长
            audio = AudioSegment.from_file(raw_generated_path)
            actual_duration_s = len(audio) / 1000.0
            # 反向计算出这次生成相当于1.0x速度下的“真实原始时长”
            actual_raw_duration_s = actual_duration_s * applied_speed
        except Exception as e:
            print(f"!! 无法获取生成音频的时长: {e}")

        # 步骤5: 时长微调 (逻辑不变)
        print("  -> [第4步] 进行高质量时长微调...")
        adjust_duration_with_rubberband(
            input_path=raw_generated_path,
            output_path=output_path_for_merge,
            target_duration_s=target_duration_s
        )
        os.remove(raw_generated_path)

        # 步骤6: 学习 (使用最关键的“真实原始时长”作为标签)
        if actual_raw_duration_s > 0:
            duration_predictor.add_data_point_and_retrain(text, actual_raw_duration_s)
    else:
        print(f"  !! API生成失败，将创建 {target_duration_s:.2f}s 的静音占位符。")
        AudioSegment.silent(duration=target_duration_s * 1000).export(output_path_for_merge, format="wav")


# ... (merge_audio 和 process_srt_files, 主函数等都保持不变) ...
def merge_audio(parsed_subtitles: List[List], current_tmp_dir: str) -> AudioSegment:
    """
    【已修复】合并指定临时目录中的音频片段。
    - 增加 current_tmp_dir 参数以接收专属临时目录。
    - 从正确的目录读取音频片段。
    """
    merged_audio = AudioSegment.empty()
    if not parsed_subtitles: return merged_audio
    print("\n--- 开始合并音频片段 ---")

    first_start_sec = time_str_to_seconds(parsed_subtitles[0][0])
    if first_start_sec > 0:
        merged_audio += AudioSegment.silent(duration=first_start_sec * 1000)

    for i, subtitle in enumerate(parsed_subtitles):
        start_time_str, end_time_str, _, _ = subtitle
        target_duration_ms = (time_str_to_seconds(end_time_str) - time_str_to_seconds(start_time_str)) * 1000

        # 【核心修正】从传入的 current_tmp_dir 而不是全局的 TMP_DIR 读取文件
        segment_file = os.path.join(current_tmp_dir, f"output_{i}.wav")

        try:
            audio_segment = AudioSegment.from_file(segment_file)
            merged_audio += audio_segment
        except Exception as e:
            print(f"!! 警告: 无法读取文件 {segment_file} ({e})。将使用 {target_duration_ms / 1000:.2f}s 的静音代替。")
            merged_audio += AudioSegment.silent(duration=target_duration_ms)

        if i < len(parsed_subtitles) - 1:
            current_end_sec = time_str_to_seconds(end_time_str)
            next_start_sec = time_str_to_seconds(parsed_subtitles[i + 1][0])
            gap_sec = next_start_sec - current_end_sec
            if gap_sec > 0.001:
                merged_audio += AudioSegment.silent(duration=gap_sec * 1000)

    return merged_audio


def process_srt_files(srt_path: str, transformers_line: int):
    srt_files_to_process = [f for f in os.listdir(srt_path) if f.lower().endswith(".srt")]

    for i, srt_file in enumerate(srt_files_to_process):
        subtitle_name, _ = os.path.splitext(srt_file)
        output_audio_file = os.path.join(OUTPUT_PATH, f"{subtitle_name}.wav")
        if os.path.exists(output_audio_file):
            print(f"文件 {output_audio_file} 已存在，跳过。")
            continue

        main_audio_path = os.path.join(REF_AUDIO_PATH, subtitle_name + ".wav")
        if not os.path.exists(main_audio_path):
            print(f"!! 警告: 找不到与 {srt_file} 匹配的参考音频文件，跳过。")
            continue

        print(f"\n[{i + 1}/{len(srt_files_to_process)}] === 处理文件 {srt_file} ===")
        print(f"[使用参考音频 {os.path.basename(main_audio_path)}]")

        current_tmp_dir = os.path.join(TMP_DIR, subtitle_name)
        os.makedirs(current_tmp_dir, exist_ok=True)

        with open(os.path.join(srt_path, srt_file), "r", encoding="utf-8") as file:
            file_content = file.read()

        parsed_subtitles = parse_subtitles(file_content, transformers_line)
        if not parsed_subtitles:
            print(f"文件 {srt_file} 未解析到有效字幕，跳过。")
            continue

        merged_subtitles = merge_consecutive_subtitles(parsed_subtitles)
        print(f"字幕数量从 {len(parsed_subtitles)}条 合并为 {len(merged_subtitles)}条。")

        # 【核心改动】检查状态文件，看是否是同一个任务的中断
        last_completed_index = -1
        status_data = load_status(STATUS_FILE)
        if status_data.get("current_file") == srt_file:
            last_completed_index = status_data.get("last_completed_index", -1)
            print(f"--- 检测到 '{srt_file}' 的中断记录，将从第 {last_completed_index + 1} 条字幕继续。 ---")
        else:
             print("--- 开始新任务或接管已放弃的任务。 ---")


        for index, subtitle in enumerate(merged_subtitles):
            if index <= last_completed_index:
                continue # 静默跳过已完成的

            # 执行TTS任务，传入专属的临时目录
            tts_process_task(index, subtitle, main_audio_path, current_tmp_dir)

            # 【核心改动】每完成一条，立即用当前任务的进度覆盖状态文件
            save_status(STATUS_FILE, srt_file, index)

        print("\n--- 所有字幕片段处理完毕，开始合并... ---")
        merged_audio = merge_audio(merged_subtitles, current_tmp_dir)
        merged_audio.export(output_audio_file, format="wav")
        print(f"\n输出最终合成音频文件：{output_audio_file}")

        # 【核心改动】任务成功完成后，清理一切痕迹
        print(f"--- '{srt_file}' 处理完成，正在清理... ---")
        # 1. 清除状态文件
        clear_status(STATUS_FILE)
        print(f"  -> 已清除状态文件。")

        # 2. 删除该任务专属的临时文件夹
        if os.path.exists(current_tmp_dir):
            shutil.rmtree(current_tmp_dir)
            print(f"  -> 已删除临时目录 '{current_tmp_dir}'。")

        print("\n--- 文件处理完毕，执行最终训练检查... ---")
        # 这是修正后的代码行
        duration_predictor.train()

    print("\n所有任务处理完毕。")


if __name__ == '__main__':
    '''f5-tts_infer-gradio'''
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
    warnings.filterwarnings("ignore", category=UserWarning, module='gradio_client.utils')
    if not os.path.exists(REF_AUDIO_PATH): os.makedirs(REF_AUDIO_PATH)
    extract_audio_from_directory(SRT_PATH, REF_AUDIO_PATH)
    process_srt_files(SRT_PATH, TRANSFORMERS_LINE)
    OUTPUT_FOLDER = os.path.join(SRT_PATH, "中配")  # <--- 推荐方式，或手动指定其他路径
    merge_audio_video(SRT_PATH, OUTPUT_FOLDER)