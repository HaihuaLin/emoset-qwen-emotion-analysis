import os
import zipfile
import subprocess
from modelscope.hub.snapshot_download import dataset_snapshot_download

def setup_intra_cloud_acceleration():
    """尝试自动检测阿里云 DSW 实例所在地域并开启内网 OSS 加速"""
    try:
        # 查询阿里云 ECS/DSW 元数据服务获取当前地域
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "2", "http://100.100.100.200/latest/meta-data/region-id"],
            capture_output=True, text=True
        )
        region = result.stdout.strip()
        if region:
            os.environ["MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION"] = region
            os.environ["INTRA_CLOUD_ACCELERATION_REGION"] = region
            print(f"[*] 检测到阿里云机房地域: {region}，已自动启用内网极速加速下载通道！")
            return region
    except Exception:
        pass
    
    # 默认回退到常用地域
    os.environ["MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION"] = "cn-hangzhou"
    print("[*] 启用默认内网加速 (cn-hangzhou)...")
    return "cn-hangzhou"

def download_and_extract_dataset(dataset_id="weisir001/EmoSet"):
    print("=" * 60)
    print(f"      开始加速下载并解压数据集: {dataset_id}")
    print("=" * 60)

    setup_intra_cloud_acceleration()

    print(f"[1/2] 正在高速下载数据集文件（11.3GB）...")
    dataset_dir = dataset_snapshot_download(
        dataset_id=dataset_id,
        revision="master"
    )
    print(f"[*] 数据集下载完成，存储路径: {dataset_dir}")

    # 检查并解压 EmoSet.zip
    zip_path = os.path.join(dataset_dir, "EmoSet.zip")
    extract_dir = os.path.join(dataset_dir, "extracted")

    if os.path.exists(zip_path):
        if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
            print(f"[2/2] 正在解压 {zip_path} 到 {extract_dir}（解压耗时约 1~2 分钟）...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("[*] 解压完成！所有图片已准备就绪。")
        else:
            print(f"[*] 发现已解压的数据目录: {extract_dir}，无需重复解压。")
    else:
        print(f"[*] 数据集无需额外解压，文件就绪。")

    print("\n✅ 数据集准备全部完成！你可以运行 python zero_shot_test.py 开始测试。")
    return dataset_dir

if __name__ == "__main__":
    download_and_extract_dataset()
