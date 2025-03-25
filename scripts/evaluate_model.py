"""
评估模型在测试集上的性能
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import torch
from src.utils.helper_functions import map_labels
import argparse

def evaluate(model, tokenizer, test_dataset, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataset:
            inputs = tokenizer(batch["clean_text"], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.tensor([map_labels(batch["label"])]).to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument(
        "--model_dir",
        default="outputs/final_model/",
        help="模型路径"
    )
    parser.add_argument(
        "--dataset_path",
        default="data/processed/dataset_dict",
        help="DatasetDict路径"
    )
    args = parser.parse_args()
    
    # 加载数据集和模型
    dataset = load_from_disk(args.dataset_path)["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    accuracy = evaluate(model, tokenizer, dataset, device)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
