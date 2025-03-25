"""
单条文本预测脚本（与之前提供的predict.py相同，但路径适配）
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from src.utils.helper_functions import clean_text
import argparse

def predict(text, model, tokenizer, device):
    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "True" if prediction == 1 else "False"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="输入文本")
    parser.add_argument("--model_dir", default="outputs/final_model/", help="模型路径")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    result = predict(args.text, model, tokenizer, device)
    print(f"输入文本：{args.text}")
    print(f"预测结果：{result}")

if __name__ == "__main__":
    main()
