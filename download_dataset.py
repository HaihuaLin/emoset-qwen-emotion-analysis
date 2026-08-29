import os
import zipfile
from modelscope import snapshot_download

# 获取当前项目根目录 (/mnt/workspace/emoset-qwen-emotion-analysis)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")

def download_and_extract_dataset(dataset_id="weisir001/EmoSet"):
    print("=" * 60)
    print(f"      使用官方标准通道下载数据集至: {DATA_DIR}")
    print("=" * 60)

    # 移除内网环境变量，使用魔搭默认官方公网通道（支持断点续传）
    os.environ.pop("MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION", None)
    os.environ.pop("INTRA_CLOUD_ACCELERATION_REGION", None)

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[1/2] 正在下载数据集文件（支持断点续传）...")
    try:
        dataset_path = snapshot_download(
            dataset_id, 
            repo_type='dataset',
            local_dir=DATA_DIR
        )
    except Exception:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
        dataset_path = dataset_snapshot_download(
            dataset_id=dataset_id,
            cache_dir=os.path.join(PROJECT_ROOT, "data")
        )

    print(f"[*] 数据集下载完成，存放路径: {dataset_path}")

    # 检查并解压 EmoSet.zip
    zip_path = os.path.join(DATA_DIR, "EmoSet.zip")
    if not os.path.exists(zip_path):
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith(".zip"):
                    zip_path = os.path.join(root, f)
                    break

    extract_dir = os.path.join(DATA_DIR, "extracted")
    if os.path.exists(zip_path):
        if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
            print(f"[2/2] 正在解压 {zip_path} 至 {extract_dir} ...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("[*] 解压完成！所有图片已准备就绪。")
        else:
            print(f"[*] 发现已解压目录: {extract_dir}，跳过解压。")

    print(f"\n✅ 数据集准备完成！存储位置: {DATA_DIR}")
    return DATA_DIR

if __name__ == "__main__":
    download_and_extract_dataset()
