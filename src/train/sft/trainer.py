"""
src/train/sft/trainer.py
职责: SFT 训练器
设计参考: GAIR-NLP DeepResearcher/verl/trainer/fsdp_sft_trainer.py
  - 使用 HuggingFace Trainer
  - 损失仅在 response tokens 上计算
  - 支持 DeepSpeed FSDP（可选）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ..data.sft_dataset import SFTDataLoader


@dataclass
class SFTConfig:
    """SFT 训练配置。"""
    output_dir: str = "./outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    max_steps: int = -1
    max_length: int = 2048
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = True
    deepspeed: Optional[str] = None  # DeepSpeed config path
    save_total_limit: int = 2


class SFTTrainer:
    """
    SFT 训练器。

    使用 HuggingFace Trainer 进行监督微调。
    损失仅在 assistant response 部分计算。

    参考 GAIR-NLP FSDPSFTTrainer:
    - loss_mask 确保仅 response tokens 参与 loss
    - 梯度累积支持大 batch
    - DeepSpeed ZeRO 支持（可选）

    使用方式:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_files="outputs/sft/sft_data.jsonl",
            config=SFTConfig(),
        )
        trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_files: str,
        eval_files: Optional[str] = None,
        config: Optional[SFTConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.train_files = train_files
        self.eval_files = eval_files
        self._setup_trainer()

    def _setup_trainer(self) -> None:
        """设置 HuggingFace Trainer。"""
        # 加载数据
        train_dataset = SFTDataLoader(
            file_path=self.train_files,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        eval_dataset = None
        if self.eval_files:
            eval_dataset = SFTDataLoader(
                file_path=self.eval_files,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            deepspeed=self.config.deepspeed,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            optim="adamw_torch",
        )

        # Collator
        data_collator = train_dataset.get_sft_collator()

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def train(self) -> None:
        """执行 SFT 训练。"""
        self.trainer.train()

    def save(self, output_dir: Optional[str] = None) -> None:
        """保存模型。"""
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def evaluate(self) -> Dict[str, float]:
        """评估模型。"""
        if self.trainer.eval_dataset is None:
            return {}
        return self.trainer.evaluate()


class SupervisedFineTrainer:
    """
    简化版 SFT 训练器（不使用 HF Trainer）。
    直接用 PyTorch 手动训练。

    适用于:
    - 调试
    - 小规模实验
    - 自定义训练循环
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_files: str,
        config: Optional[SFTConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataset = SFTDataLoader(
            file_path=train_files,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

    def train(self) -> None:
        """手动训练循环。"""
        from torch.utils.data import DataLoader
        from torch.optim import AdamW

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.get_sft_collator(),
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.model.train()
        global_step = 0
        total_loss = 0.0

        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        print(f"Step {global_step}: loss={avg_loss:.4f}")
                        total_loss = 0.0

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        self.save(f"{self.config.output_dir}/checkpoint-{global_step}")

        self.save(f"{self.config.output_dir}/final")

    def save(self, output_dir: str) -> None:
        """保存模型。"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
