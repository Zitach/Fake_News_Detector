from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model_and_peft(model_name):
    """
    加载Phi-4-Mini模型并配置LoRA
    """
    # 加载预训练模型并适配为分类任务
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,          # 二分类（True/False）
        problem_type="single_label_classification"
    )
    
    # 配置LoRA参数
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=8,                  # 隐层维度
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]  # 可选：指定要适配的模块
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
