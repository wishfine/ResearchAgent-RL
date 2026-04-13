"""
ResearchAgent-RL Policy
========================
Policy 接口和实现。

MVP 阶段提供：
1. RandomPolicy（随机 baseline）
2. RuleBasedPolicy（规则 baseline）
3. LLMPolicy（基于 LLM 的 policy，SFT/RL 训练后使用）
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import random
import json

from data.schemas import Observation, Action, ToolName, TaskType
from tools.base import ToolRegistry


# =============================================================================
# Base Policy
# =============================================================================

class BasePolicy(ABC):
    """Policy 基类"""

    @abstractmethod
    def predict(self, obs: Observation) -> Action:
        """
        给定 observation，输出 action
        """
        pass

    @abstractmethod
    def reset(self):
        """重置 policy 状态（如有内部状态）"""
        pass


# =============================================================================
# Random Policy
# =============================================================================

class RandomPolicy(BasePolicy):
    """随机 Policy，用于 baseline 和测试"""

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def reset(self):
        pass

    def predict(self, obs: Observation) -> Action:
        """随机选择一个 tool 和参数"""
        available_tools = self.tool_registry.list_tool_names()

        # 倾向于选择 ANSWER 如果步数不多
        if obs.remaining_steps <= 3 and ToolName.ANSWER in available_tools:
            tool = ToolName.ANSWER
        else:
            tool = random.choice([t for t in available_tools if t != ToolName.ANSWER])

        params = self._generate_random_params(tool, obs)

        return Action(
            tool=tool,
            intent=f"Random {tool.value}",
            params=params,
            reasoning="Random policy selection"
        )

    def _generate_random_params(self, tool: ToolName, obs: Observation) -> Dict[str, Any]:
        """为不同 tool 生成随机参数"""
        if tool == ToolName.SEARCH:
            queries = [
                obs.user_query,
                obs.current_evidence_gap,
                obs.current_subgoal
            ]
            return {
                "query": random.choice([q for q in queries if q]) or "research query",
                "topk": random.randint(3, 5),
                "target_gap": obs.current_evidence_gap,
                "query_rationale": "random search"
            }

        elif tool == ToolName.READ:
            candidates = [c.chunk_id for c in obs.candidate_chunks]
            if candidates:
                n = min(random.randint(1, 3), len(candidates))
                return {
                    "chunk_ids": random.sample(candidates, n),
                    "read_goal": "read for evidence"
                }
            return {"chunk_ids": [], "read_goal": ""}

        elif tool == ToolName.RERANK:
            candidates = [c.chunk_id for c in obs.candidate_chunks]
            if candidates:
                return {
                    "query": obs.user_query,
                    "candidate_chunk_ids": candidates,
                    "topk": min(5, len(candidates)),
                    "rerank_goal": "prioritize relevant"
                }
            return {"query": obs.user_query, "candidate_chunk_ids": [], "topk": 5, "rerank_goal": ""}

        elif tool == ToolName.CITE:
            read_ids = [s.chunk_id for s in obs.read_summaries]
            if read_ids:
                return {
                    "chunk_ids": random.sample(read_ids, min(3, len(read_ids))),
                    "claims": ["supporting claim"] * min(3, len(read_ids))
                }
            return {"chunk_ids": [], "claims": []}

        elif tool == ToolName.ANSWER:
            return {
                "answer_text": f"This is a mock answer for: {obs.user_query[:50]}...",
                "cited_chunk_ids": list(obs.cited_chunks)
            }

        return {}


# =============================================================================
# Rule-Based Policy
# =============================================================================

class RuleBasedPolicy(BasePolicy):
    """
    基于规则的 Policy
    按照固定策略执行 SEARCH → READ → RERANK → CITE → ANSWER

    这是 SFT 和 RL 的 baseline，用于：
    1. 生成训练轨迹
    2. 提供可运行的最低 baseline
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self._step_count = 0

    def reset(self):
        self._step_count = 0

    def predict(self, obs: Observation) -> Action:
        """
        基于规则的策略：
        1. 先 SEARCH 获取候选
        2. READ 已检索的候选获取内容
        3. 必要时 RERANK
        4. CITE 声明引用
        5. ANSWER 生成答案
        """
        step = self._step_count
        self._step_count += 1

        # 根据已有信息决定下一步
        if not obs.search_history:
            # 第一步：搜索
            return self._make_search_action(obs)

        elif not obs.read_summaries:
            # 第二步：读取所有候选
            return self._make_read_action(obs)

        elif obs.remaining_steps <= 3:
            # 剩余步数少，收尾
            return self._make_answer_action(obs)

        elif len(obs.search_history) < 2 and len(obs.read_summaries) < 5:
            # 可以再搜索
            return self._make_search_action(obs)

        elif obs.cited_chunks:
            # 有引用，可以回答
            return self._make_answer_action(obs)

        else:
            # 默认读取更多
            return self._make_read_action(obs)

    def _make_search_action(self, obs: Observation) -> Action:
        query = obs.user_query if obs.user_query else obs.current_evidence_gap
        if not query:
            query = "research"

        return Action(
            tool=ToolName.SEARCH,
            intent="Retrieve relevant document chunks",
            params={
                "query": query,
                "topk": 5,
                "target_gap": obs.current_evidence_gap,
                "query_rationale": "initial search for task"
            },
            reasoning="Rule-based: initial search"
        )

    def _make_read_action(self, obs: Observation) -> Action:
        # 读取尚未读取的候选
        read_ids = {s.chunk_id for s in obs.read_summaries}
        to_read = [c.chunk_id for c in obs.candidate_chunks if c.chunk_id not in read_ids]

        if not to_read:
            # 如果没有候选，先搜索
            return self._make_search_action(obs)

        return Action(
            tool=ToolName.READ,
            intent="Read retrieved chunks for evidence",
            params={
                "chunk_ids": to_read[:5],
                "read_goal": f"find evidence for: {obs.user_query}"
            },
            reasoning="Rule-based: read chunks for evidence"
        )

    def _make_rerank_action(self, obs: Observation) -> Action:
        return Action(
            tool=ToolName.RERANK,
            intent="Prioritize most relevant chunks",
            params={
                "query": obs.user_query,
                "candidate_chunk_ids": [c.chunk_id for c in obs.candidate_chunks],
                "topk": 5,
                "rerank_goal": "prioritize evidence quality"
            },
            reasoning="Rule-based: rerank candidates"
        )

    def _make_cite_action(self, obs: Observation) -> Action:
        read_ids = [s.chunk_id for s in obs.read_summaries]
        return Action(
            tool=ToolName.CITE,
            intent="Cite supporting chunks",
            params={
                "chunk_ids": read_ids[:5],
                "claims": [f"Claim for chunk {cid}" for cid in read_ids[:5]]
            },
            reasoning="Rule-based: cite evidence"
        )

    def _make_answer_action(self, obs: Observation) -> Action:
        # 构建答案
        read_summaries = obs.read_summaries

        answer_parts = [f"Based on the research:\n\n"]
        for s in read_summaries[:3]:
            answer_parts.append(f"- {s.summary[:200]}\n")

        answer_parts.append(f"\nConclusion for: {obs.user_query}")

        return Action(
            tool=ToolName.ANSWER,
            intent="Generate final answer with citations",
            params={
                "answer_text": "".join(answer_parts),
                "cited_chunk_ids": list(set(s.chunk_id for s in read_summaries))
            },
            reasoning="Rule-based: generate answer"
        )


# =============================================================================
# LLM Policy (SFT/RL trained)
# =============================================================================

class LLMPolicy(BasePolicy):
    """
    基于 LLM 的 Policy

    使用 Hugging Face Transformers + PEFT(LoRA) 加载训练好的模型
    输入：Observation 的 prompt 序列化
    输出：Action（JSON 格式）

    MVP 阶段该类为占位，后续：
    1. Stage 2 实现 SFT warm-up 后的模型加载
    2. Stage 3 实现 veRL 训练后的模型加载
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.tool_registry = tool_registry
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载训练好的模型"""
        # TODO: Stage 2 实现
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        pass

    def reset(self):
        pass

    def predict(self, obs: Observation) -> Action:
        """
        使用 LLM 预测 action
        MVP 阶段 fallback 到 RandomPolicy
        """
        if self.model is None:
            # Fallback to random policy for MVP
            random_policy = RandomPolicy(self.tool_registry)
            return random_policy.predict(obs)

        # TODO: 真正的 LLM 推理
        raise NotImplementedError("LLM inference not implemented yet")


# =============================================================================
# Policy Wrapper (用于训练框架)
# =============================================================================

class PolicyWrapper:
    """
    统一 Policy 接口，用于适配不同训练框架

    提供：
    - compute_action(obs) -> action
    - compute_log_prob(obs, action) -> log_prob
    - get_policy_state_dict() -> dict
    - load_policy_state_dict(state_dict)
    """

    def __init__(self, policy: BasePolicy, device: str = "cuda"):
        self.policy = policy
        self.device = device

    def compute_action(self, obs: Observation, deterministic: bool = False) -> Action:
        """计算 action"""
        return self.policy.predict(obs)

    def evaluate_actions(self, obs: Observation, action: Action) -> Dict[str, float]:
        """
        评估 action，返回 log_prob 等指标
        用于 PPO 等 on-policy 算法
        """
        # MVP 返回默认值
        return {"log_prob": 0.0, "entropy": 0.0, "value": 0.0}

    def get_policy_state_dict(self) -> Dict[str, Any]:
        """获取 policy 参数"""
        if hasattr(self.policy, 'model') and self.policy.model is not None:
            return self.policy.model.state_dict()
        return {}

    def load_policy_state_dict(self, state_dict: Dict[str, Any]):
        """加载 policy 参数"""
        if hasattr(self.policy, 'model') and self.policy.model is not None:
            self.policy.model.load_state_dict(state_dict)
