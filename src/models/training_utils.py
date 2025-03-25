from transformers import TrainingArguments, Trainer

def get_trainer(model, tokenizer, dataset, output_dir):
    """
    配置Trainer并返回实例
    """
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 根据GPU显存调整
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard"
    )

    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=lambda pred: {"accuracy": (pred.predictions.argmax(-1) == pred.label_ids).mean()}
    )
    
    return trainer
