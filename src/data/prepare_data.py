import pandas as pd
from datasets import DatasetDict
from src.utils.helper_functions import map_labels, clean_text

def prepare_dataset():
    """
    加载和预处理合并后的数据集，返回DatasetDict格式数据。
    """
    # 加载处理后的合并数据集
    merged_df = pd.read_csv("data/processed/merged_dataset.csv")
    
    # 清洗文本并转换标签
    merged_df["clean_text"] = merged_df["statement"].apply(clean_text)
    merged_df["label"] = merged_df["label"].apply(map_labels)
    
    # 拆分训练集和测试集（8:2）
    train_test = merged_df.sample(frac=1).reset_index(drop=True)  # 打乱数据
    train_df = train_test[:int(0.8 * len(train_test))]
    test_df = train_test[int(0.8 * len(train_test)):]
    
    # 转换为HuggingFace Dataset格式
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    
    return dataset
