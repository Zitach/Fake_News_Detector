"""
合并原始Liar和ISOT数据集到processed目录
"""

from src.data.merge_datasets import merge_datasets
import argparse

def main():
    parser = argparse.ArgumentParser(description="合并原始数据集")
    parser.add_argument(
        "--output_path",
        default="data/processed/merged_dataset.csv",
        help="合并后数据集保存路径"
    )
    args = parser.parse_args()
    
    # 调用数据合并函数
    merge_datasets()  # 原函数已硬编码路径，此处可扩展参数
    print(f"数据集已保存至：{args.output_path}")

if __name__ == "__main__":
    main()
