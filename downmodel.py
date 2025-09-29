import json
import os
import sys

import time
from pathlib import Path
from huggingface_hub import hf_hub_download
def download_model_files():
    """
    从Hugging Face Hub下载IndexTTS-1.5模型所需的文件。
    """
    repo_id = "IndexTeam/IndexTTS-1.5"
    local_dir = "checkpoints"
    
    # 确保本地目录存在
    if not os.path.exists(local_dir):
        print(f"创建目录: {local_dir}")
        os.makedirs(local_dir)

    # 需要下载的文件列表
    files_to_download = [
        "config.yaml",
        "bigvgan_discriminator.pth",
        "bigvgan_generator.pth",
        "bpe.model",
        "dvae.pth",
        "gpt.pth",
        "unigram_12000.vocab"
    ]
    
    is_bigvgan_discriminator=Path(f'./{local_dir}/bigvgan_discriminator.pth').exists()
    for filename in files_to_download:
        # 检查文件是否已存在，如果存在则跳过下载
        is_exists = Path(f'{local_dir}/{filename}').exists()
        if is_exists:
            if filename !='config.yaml' or is_bigvgan_discriminator:
                # 如果 config.yaml 已存在，但不存在 bigvgan_discriminator.pth，此时需重新下载 config.yaml
                # 否则跳过
                print(f"文件 {filename} 已存在，跳过下载。")
                continue
            
        print(f"正在下载 {filename} 到 {local_dir}...")
        try:
            # 使用hf_hub_download下载文件
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                # resume_download=True  # 如果需要，可以开启断点续传
            )
            print(f"下载 {filename} 完成。")
        except Exception as e:
            print(f"下载 {filename} 失败: {e}")
            # 你可以在这里决定是继续下载其他文件还是中止程序
            # return False # 如果希望下载失败时中止，可以取消这行注释
    
    for filename in files_to_download:
        # 检查文件是否已存在，如果存在则跳过下载
        local_file_path = Path(f'./{local_dir}/{filename}')
        if not local_file_path.exists() or local_file_path.stat().st_size==0:
            print(f"文件 {filename} 不存在或尺寸为0，请保证网络连接正常，然后删掉该文件后重新启动下载。")
            return False
    
    print("所有模型文件下载检查完成！\n")
    return True

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'


print("\n----正在检查是否已下载 IndexTTS-1.5 模型...")
download_success = download_model_files()

if not download_success:
    print("\n\n############模型文件下载失败，请检查网络连接或手动下载。程序即将退出。\n")
    time.sleep(5)
    sys.exit() # 如果模型是必需的，可以选择在这里退出

# 下载完成后，继续启动WebUI
print("\n模型文件准备就绪，正在启动 WebUI...")
print("\n\n********请等待启动完成，显示\" Running on local URL:  http://127.0.0.1:7860 \"时，在浏览器里打开该地址********\n\n")
