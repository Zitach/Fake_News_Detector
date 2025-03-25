from src.data.prepare_data import prepare_dataset
from src.models.model_utils import load_model_and_peft
from src.models.training_utils import get_trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/phi-4-mini")
    parser.add_argument("--output_dir", default="outputs/final_model/")
    args = parser.parse_args()
    
    # 数据准备
    dataset = prepare_dataset()
    
    # 模型和PEFT配置
    model, tokenizer = load_model_and_peft(args.model_name)
    
    # 初始化Trainer并训练
    trainer = get_trainer(model, tokenizer, dataset, args.output_dir)
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
