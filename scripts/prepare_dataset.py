"""
将合并后的数据集拆分为训练集和测试集（HuggingFace Dataset格式）
"""

from src.data.prepare_data import prepare_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="准备训练/测试数据集")
    parser.add_argument(
        "--output_dir",
        default="data/processed/",
        help="保存DatasetDict的目录"
    )
    args = parser.parse_args()
    
    dataset = prepare_dataset()
    # 保存DatasetDict（可选）
    dataset.save_to_disk(f"{args.output_dir}/dataset_dict")
    print("数据集已保存")

if __name__ == "__main__":
    main()
