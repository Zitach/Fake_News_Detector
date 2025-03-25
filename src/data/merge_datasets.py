import pandas as pd
from src.utils.helper_functions import map_labels

def merge_datasets():
    """
    合并原始Liar和ISOT数据集，生成最终的merged_dataset.csv
    """
    # 加载Liar数据集
    liar_train = pd.read_csv("data/raw/liar_dataset/train.tsv", sep='\t')
    liar_valid = pd.read_csv("data/raw/liar_dataset/valid.tsv", sep='\t')
    liar_test = pd.read_csv("data/raw/liar_dataset/test.tsv", sep='\t')
    liar_full = pd.concat([liar_train, liar_valid, liar_test])
    
    # 处理Liar标签：将['true', 'mostly-true', ...]映射为'True'/'False'
    liar_full["label"] = liar_full["label"].apply(lambda x: "True" if x in ['true', 'mostly-true'] else "False")
    
    # 加载ISOT数据集
    isot_true = pd.read_csv("data/raw/isot_dataset/True.csv")
    isot_fake = pd.read_csv("data/raw/isot_dataset/Fake.csv")
    isot_true["label"] = "True"
    isot_fake["label"] = "False"
    isot_full = pd.concat([isot_true, isot_fake])
    
    # 合并两个数据集
    combined_df = pd.concat([liar_full[["statement", "label"]], 
                            isot_full[["statement", "label"]]]).reset_index(drop=True)
    
    # 保存到processed目录
    combined_df.to_csv("data/processed/merged_dataset.csv", index=False)
    print("Merged dataset saved to data/processed/merged_dataset.csv")

if __name__ == "__main__":
    merge_datasets()
