"""
生成评估报告（如混淆矩阵、分类报告）
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import argparse

def generate_report(model, tokenizer, test_dataset, device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataset:
            inputs = tokenizer(batch["clean_text"], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = [int(batch["label"])]  # 原始标签（0/1）
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 生成报告
    report = classification_report(all_labels, all_preds, target_names=["False", "True"])
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/final_model/")
    parser.add_argument("--dataset_path", default="data/processed/dataset_dict")
    args = parser.parse_args()
    
    dataset = load_from_disk(args.dataset_path)["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    report, cm = generate_report(model, tokenizer, dataset, device)
    print("分类报告：")
    print(report)
    print("\n混淆矩阵：")
    print(cm)

if __name__ == "__main__":
    main()
