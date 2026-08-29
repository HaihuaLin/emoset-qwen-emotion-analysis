import os
import zipfile
import subprocess
from modelscope import snapshot_download

# 获取当前项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")

def setup_intra_cloud_acceleration():
    """尝试自动检测阿里云 DSW 实例所在地域并开启内网 OSS 加速"""
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "2", "http://100.100.100.200/latest/meta-data/region-id"],
            capture_output=True, text=True
        )
        region = result.stdout.strip()
        if region:
            os.environ["MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION"] = region
            os.environ["INTRA_CLOUD_ACCELERATION_REGION"] = region
            print(f"[*] 检测到阿里云地域: {region}，已自动启用内网极速加速！")
            return region
    except Exception:
        pass
    
    os.environ["MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION"] = "cn-hangzhou"
    print("[*] 启用默认内网加速 (cn-hangzhou)...")
    return "cn-hangzhou"

def download_and_extract_dataset(dataset_id="weisir001/EmoSet"):
    print("=" * 60)
    print(f"      下载并解压数据集至项目目录: {DATA_DIR}")
    print("=" * 60)

    setup_intra_cloud_acceleration()
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[1/2] 正在将数据集下载至: {DATA_DIR} ...")
    # 使用 local_dir / cache_dir 确保直接保存到项目目录
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

    print(f"[*] 数据集下载完成，位置: {dataset_path}")

    # 查找并解压 EmoSet.zip
    zip_path = os.path.join(DATA_DIR, "EmoSet.zip")
    if not os.path.exists(zip_path):
        # 递归寻找 zip 文件
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
            print("[*] 解压完成！所有图片数据已就绪。")
        else:
            print(f"[*] 发现已解压目录: {extract_dir}，跳过解压。")

    print(f"\n✅ 数据集已完整保存在项目目录下: {DATA_DIR}")
    return DATA_DIR

if __name__ == "__main__":
    download_and_extract_dataset()
