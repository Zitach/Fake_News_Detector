from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from src.utils.helper_functions import clean_text
import argparse

def predict(text, model, tokenizer, device):
    """
    对单条文本进行预测并返回结果（True/False）
    """
    # 预处理输入文本
    cleaned_text = clean_text(text)
    
    # 生成输入张量
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 根据模型最大长度调整
    ).to(device)
    
    # 禁用梯度计算以加速推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测结果（0: False, 1: True）
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "True" if prediction == 1 else "False"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="需要预测的文本内容"
    )
    parser.add_argument(
        "--model_dir",
        default="outputs/final_model/",
        help="模型保存路径（默认为最终训练好的模型目录）"
    )
    args = parser.parse_args()
    
    # 设备选择（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # 将模型移动到设备并设置为评估模式
    model.to(device)
    model.eval()
    
    # 执行预测
    result = predict(args.text, model, tokenizer, device)
    print(f"输入文本：{args.text}")
    print(f"预测结果：{result}")

if __name__ == "__main__":
    main()
