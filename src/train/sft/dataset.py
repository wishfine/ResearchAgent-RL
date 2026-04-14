"""
src/train/sft/dataset.py
职责: 将 EpisodeResult trajectory 转换为 SFT 可用格式
设计参考: GAIR-NLP DeepResearcher/verl/utils/dataset/sft_dataset.py
  - 使用 role-based chat format (user/assistant)
  - response 部分计算 loss，prompt 部分不计算
  - 损失掩码仅在 response tokens 上

格式:
  conversations = [
    {"role": "system", "content": "你是一个科研助手..."},
    {"role": "user", "content": "任务描述 + 初始 observation"},
    {"role": "assistant", "content": "action with reasoning"},
    {"role": "user", "content": "observation after action"},
    {"role": "assistant", "content": "next action"},
    ...
    {"role": "assistant", "content": "final answer"},
  ]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import os

from ...schema.result import EpisodeResult
from ...schema.action import TrajectoryStep
from ...schema.observation import Observation


class SFTRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class SFTExample:
    """
    单条 SFT 训练样本。
    conversations: List[{"role": str, "content": str}]
    loss_mask: List[int] - 1 表示该 token 参与 loss 计算
    """
    conversations: List[Dict[str, str]]
    loss_mask: List[int] = field(default_factory=list)
    task_id: str = ""
    final_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "conversations": self.conversations,
            "loss_mask": self.loss_mask,
            "task_id": self.task_id,
            "final_answer": self.final_answer,
        }


class TrajectoryToSFTConverter:
    """
    将 EpisodeResult trajectory 转换为 SFT 格式。

    参考 GAIR-NLP:
    - System prompt 定义 agent 角色和工具集合
    - User: 初始 task + 每步 observation
    - Assistant: 每步 action + reasoning
    - Final answer 在最后一条 assistant message

    损失计算:
    - 仅在 assistant 的 content 上计算 loss
    - user 和 system 的 content 不参与 loss
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_reasoning: bool = True,
    ):
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.include_reasoning = include_reasoning

    def _default_system_prompt(self) -> str:
        return """你是一个科研任务求解 Agent。
你有以下工具可用：
- SEARCH: 搜索相关文档 chunks
- READ: 阅读指定 chunks，提取关键证据
- RERANK: 对候选 chunks 重排序
- CITE: 引用 chunks 支持 claims
- ANSWER: 提交最终答案

每次回复请先给出 reasoning（<reasoning>...</reasoning>），然后给出 action（<action>...</action>）。
当有足够证据时，提交 ANSWER action 结束任务。"""

    def convert(self, episode_result: EpisodeResult) -> SFTExample:
        """
        将单个 EpisodeResult 转换为 SFTExample。

        Args:
            episode_result: 环境返回的完整 episode 记录

        Returns:
            SFTExample，包含 conversation 格式和 loss mask
        """
        conversations = []

        # System message
        conversations.append({
            "role": SFTRole.SYSTEM.value,
            "content": self.system_prompt,
        })

        # 获取第一条 observation（从 trajectory 推断）
        task_id = episode_result.task_id
        final_answer = episode_result.final_answer

        # 解析 trajectory，构建 conversation
        for step in episode_result.trajectory:
            action = step.action

            # User message: 描述当前 observation 状态（如果是第一步则为初始任务）
            if step.step_idx == 0:
                user_content = f"任务 (task_id={task_id}):\n"
                user_content += f"请帮我完成以下任务：\n"
                user_content += f"基于提供的文档库，检索、阅读并综合证据，给出答案。\n"
                user_content += f"当前状态：无候选文档，需要开始检索。"
            else:
                # 非第一步：描述上一步执行结果
                prev_action = episode_result.trajectory[step.step_idx - 1].action
                user_content = f"上一步执行了 {prev_action.tool}，结果：\n"
                if step.tool_result is not None:
                    if hasattr(step.tool_result, "summary"):
                        user_content += step.tool_result.summary()
                    elif hasattr(step.tool_result, "data"):
                        result_data = step.tool_result.data
                        if isinstance(result_data, dict):
                            user_content += f"返回 {len(result_data)} 条结果"
                user_content += f"\n\n当前 remaining_steps={episode_result.total_steps - step.step_idx}"

            conversations.append({
                "role": SFTRole.USER.value,
                "content": user_content,
            })

            # Assistant message: action + reasoning
            assistant_content = self._format_action(action)
            conversations.append({
                "role": SFTRole.ASSISTANT.value,
                "content": assistant_content,
            })

        # 如果 final_answer 与最后一条 assistant 不同，追加 final answer
        if final_answer and final_answer not in conversations[-1]["content"]:
            conversations.append({
                "role": SFTRole.USER.value,
                "content": "请基于以上证据，给出最终答案。",
            })
            conversations.append({
                "role": SFTRole.ASSISTANT.value,
                "content": f"最终答案：\n{final_answer}",
            })

        return SFTExample(
            conversations=conversations,
            task_id=task_id,
            final_answer=final_answer,
        )

    def _format_action(self, action) -> str:
        """将 Action 格式化为字符串。"""
        if self.include_reasoning and action.reasoning:
            content = f"<reasoning>{action.reasoning}</reasoning>\n"
        else:
            content = ""

        content += f"<action>\n"
        content += f"tool: {action.tool}\n"
        content += f"intent: {action.intent}\n"

        # 格式化 params
        for key, value in action.params.items():
            if key == "answer_text":
                # answer 特殊处理：截断过长的 answer
                display_value = value[:500] + "..." if len(str(value)) > 500 else value
            else:
                display_value = value
            content += f"{key}: {display_value}\n"

        content += f"</action>"
        return content

    def convert_batch(
        self, episode_results: List[EpisodeResult]
    ) -> List[SFTExample]:
        """批量转换。"""
        return [self.convert(ep) for ep in episode_results]


class SFTDatasetWriter:
    """
    将 SFTExample 写入文件（JSONL 格式）。
    参考 GAIR-NLP 的 parquet 输出格式，简化为 JSONL。
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write(
        self,
        examples: List[SFTExample],
        filename: str = "sft_data.jsonl",
    ) -> str:
        """
        写入 JSONL 文件。

        每行格式:
        {"conversations": [...], "task_id": "...", "final_answer": "...", "loss_mask": [...]}
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        return filepath

    def write_parquet(
        self,
        examples: List[SFTExample],
        filename: str = "sft_data.parquet",
    ) -> str:
        """
        写入 Parquet 文件（需要 pandas + pyarrow）。
        参考 GAIR-NLP 的 parquet 格式。
        """
        try:
            import pandas as pd
            records = [ex.to_dict() for ex in examples]
            df = pd.DataFrame(records)
            filepath = os.path.join(self.output_dir, filename)
            df.to_parquet(filepath, index=False)
            return filepath
        except ImportError:
            # fallback 到 JSONL
            return self.write(examples, filename.replace(".parquet", ".jsonl"))


class SFTDataLoader:
    """
    加载 SFT 数据，供 HF Trainer 使用。
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[SFTExample] = []
        self._load()

    def _load(self) -> None:
        """加载 JSONL 文件。"""
        self.examples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    convs = data["conversations"]
                    # 重建 SFTExample
                    example = SFTExample(
                        conversations=convs,
                        loss_mask=data.get("loss_mask", []),
                        task_id=data.get("task_id", ""),
                        final_answer=data.get("final_answer", ""),
                    )
                    self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回单条样本，格式化为 HF Trainer 可用 dict。

        返回:
        {
            "input_ids": List[int],
            "attention_mask": List[int],
            "labels": List[int],  # -100 表示不参与 loss
        }
        """
        example = self.examples[idx]

        # 将 conversations 格式化为字符串
        # 使用简单格式：每条 message 用 \n\n 分隔
        text = self._format_conversations(example.conversations)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 构建 labels: 仅 assistant 部分参与 loss
        labels = [-100] * len(input_ids)

        # 找到 assistant 内容的位置，标记为参与 loss
        # 简化处理：假设 assistant 内容在对应位置
        # 实际应用中需要更精确的定位
        labels = self._compute_labels(input_ids, example, text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """将 conversations 格式化为字符串。"""
        lines = []
        for msg in conversations:
            role = msg["role"]
            content = msg["content"]
            lines.append(f"{role.upper()}: {content}")
        return "\n\n".join(lines)

    def _compute_labels(
        self, input_ids: List[int], example: SFTExample, text: str
    ) -> List[int]:
        """
        计算 labels。
        仅 assistant 内容参与 loss，user 和 system 内容设为 -100。

        简化实现：找到所有 "ASSISTANT:" 的位置，其后内容参与 loss。
        """
        labels = [-100] * len(input_ids)

        text_lower = text.lower()
        assistant_marker = "assistant:".encode().decode('utf-8', errors='ignore')

        # 找到所有 assistant 位置
        import re
        for match in re.finditer(r'assistant:', text_lower):
            start_pos = match.start()
            # 找到对应在 input_ids 中的位置
            # 简化：向后查找直到下一个 role marker
            end_pattern = r'\n\n(system:|user:|assistant:)'
            end_match = re.search(end_pattern, text_lower[start_pos + len(assistant_marker):])
            if end_match:
                end_pos = start_pos + len(assistant_marker) + end_match.start()
            else:
                end_pos = len(text_lower)

            # 转换为 token 位置（简化处理）
            # 实际需要字符位置到 token 位置的映射
            # 这里用近似方法
            try:
                assistant_text = text_lower[start_pos:end_pos]
                assistant_encoded = self.tokenizer(
                    assistant_text,
                    add_special_tokens=False,
                    return_tensors=None,
                )
                assistant_tokens = assistant_encoded["input_ids"]

                # 找到这些 token 在 input_ids 中的位置
                for tok in assistant_tokens:
                    try:
                        tok_pos = input_ids.index(tok)
                        labels[tok_pos] = tok
                    except ValueError:
                        pass
            except Exception:
                pass

        return labels

    def get_sft_collator(self):
        """
        返回 HF DataCollator。
        """
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,  # 不需要 model，仅用于 padding
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
