"""
ResearchAgent-RL Evaluator
===========================
双层评测系统：
1. 结果指标（outcome metrics）
2. 过程指标（process metrics）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from data.schemas import TaskSample, EpisodeResult, TerminationType, TaskType, Observation


# =============================================================================
# 结果指标（Outcome Metrics）
# =============================================================================

@dataclass
class OutcomeMetrics:
    """第一层：结果指标"""
    task_id: str
    task_success: bool               # 任务是否成功完成
    answer_quality: float            # [0, 1] 答案质量
    rubric_completion: float          # [0, 1] Rubric 完成度
    citation_recall: float            # [0, 1] 引用召回率
    citation_precision: float         # [0, 1] 引用精确率

    # 汇总分数
    def mean_quality(self) -> float:
        return (self.answer_quality + self.rubric_completion) / 2

    def citation_f1(self) -> float:
        if self.citation_recall + self.citation_precision == 0:
            return 0.0
        return 2 * self.citation_recall * self.citation_precision / (self.citation_recall + self.citation_precision)


# =============================================================================
# 过程指标（Process Metrics）
# =============================================================================

@dataclass
class ProcessMetrics:
    """第二层：过程指标"""
    task_id: str
    total_steps: int                 # 总步数
    search_count: int                # SEARCH 次数
    read_count: int                  # READ 次数
    rerank_count: int                # RERANK 次数
    cite_count: int                  # CITE 次数
    invalid_action_rate: float        # 非法 action 比例
    repeated_query_rate: float        # 重复查询比例
    repeated_read_rate: float        # 重复读取比例
    mean_new_evidence_per_read: float # 每次 READ 平均新证据数
    no_progress_termination: bool     # 是否因 no_progress 终止
    termination: TerminationType       # 终止类型


# =============================================================================
# Aggregated Metrics
# =============================================================================

@dataclass
class AggregatedMetrics:
    """聚合后的评测指标"""
    # 结果指标汇总
    task_success_rate: float
    mean_answer_quality: float
    mean_rubric_completion: float
    mean_citation_recall: float
    mean_citation_precision: float
    mean_citation_f1: float

    # 过程指标汇总
    mean_steps: float
    mean_search_count: float
    mean_read_count: float
    invalid_action_rate: float
    repeated_query_rate: float
    repeated_read_rate: float
    no_progress_termination_rate: float

    # 详细记录
    per_task_outcome: List[OutcomeMetrics]
    per_task_process: List[ProcessMetrics]


# =============================================================================
# Case Study & Failure Analysis
# =============================================================================

@dataclass
class CaseStudy:
    """Case Study 模板"""
    task_id: str
    task_type: TaskType
    user_query: str
    trajectory_summary: str
    key_decisions: List[str]          # 关键决策点
    evidence_used: List[str]          # 使用了哪些证据
    final_answer_preview: str         # 答案预览（前200字）
    termination_reason: str
    outcome_metrics: OutcomeMetrics
    process_metrics: ProcessMetrics
    strength: str                     # 亮点
    weakness: str                      # 不足


@dataclass
class FailureCase:
    """Failure Case 模板"""
    task_id: str
    failure_type: str                 # "citation_miss", "irrelevant_retrieval", "premature_answer", etc.
    root_cause: str                    # 根本原因分析
    trajectory_excerpt: str            # 关键 trajectory 片段
    expected_action: str               # 期望 action
    actual_action: str                 # 实际 action
    reward_breakdown: Dict[str, float] # reward 分解


# =============================================================================
# Main Evaluator
# =============================================================================

class Evaluator:
    """
    评测器

    使用方法：
    evaluator = Evaluator(task_dataset)
    results = evaluator.evaluate(policy, split="dev")
    report = evaluator.generate_report(results)
    """

    def __init__(self, tasks: List[TaskSample]):
        self.tasks = {t.task_id: t for t in tasks}

    def evaluate(
        self,
        policy,  # BasePolicy
        split: str = "dev",
        max_samples: Optional[int] = None
    ) -> List[EpisodeResult]:
        """
        对 policy 在任务集上做评测
        返回每个 task 的 EpisodeResult
        """
        from env.environment import ResearchAgentEnv
        from tools.base import ToolRegistry

        results = []
        task_list = [t for t in self.tasks.values()]

        # 可以按 split 过滤（简化实现，假设所有 task 都在一个池）
        if max_samples:
            task_list = task_list[:max_samples]

        for task in task_list:
            env = ResearchAgentEnv()
            env.reset(task)

            obs = env.current_obs
            policy.reset()

            while True:
                action = policy.predict(obs)
                obs, reward, done, info = env.step(action)

                if done:
                    break

            # finalize episode
            result = env.finalize_episode()
            results.append(result)

        return results

    def compute_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> AggregatedMetrics:
        """计算聚合指标"""
        outcome_list = []
        process_list = []

        for ep in episodes:
            task = self.tasks.get(ep.task_id)
            if not task:
                continue

            # 结果指标
            outcome = self._compute_outcome_metrics(task, ep)
            outcome_list.append(outcome)

            # 过程指标
            process = self._compute_process_metrics(ep)
            process_list.append(process)

        # 聚合
        n = len(outcome_list)
        if n == 0:
            return AggregatedMetrics(
                task_success_rate=0, mean_answer_quality=0, mean_rubric_completion=0,
                mean_citation_recall=0, mean_citation_precision=0, mean_citation_f1=0,
                mean_steps=0, mean_search_count=0, mean_read_count=0,
                invalid_action_rate=0, repeated_query_rate=0, repeated_read_rate=0,
                no_progress_termination_rate=0,
                per_task_outcome=[], per_task_process=[]
            )

        return AggregatedMetrics(
            task_success_rate=sum(1 for o in outcome_list if o.task_success) / n,
            mean_answer_quality=sum(o.answer_quality for o in outcome_list) / n,
            mean_rubric_completion=sum(o.rubric_completion for o in outcome_list) / n,
            mean_citation_recall=sum(o.citation_recall for o in outcome_list) / n,
            mean_citation_precision=sum(o.citation_precision for o in outcome_list) / n,
            mean_citation_f1=sum(o.mean_quality() for o in outcome_list) / n,
            mean_steps=sum(p.total_steps for p in process_list) / n,
            mean_search_count=sum(p.search_count for p in process_list) / n,
            mean_read_count=sum(p.read_count for p in process_list) / n,
            invalid_action_rate=sum(p.invalid_action_rate for p in process_list) / n,
            repeated_query_rate=sum(p.repeated_query_rate for p in process_list) / n,
            repeated_read_rate=sum(p.repeated_read_rate for p in process_list) / n,
            no_progress_termination_rate=sum(1 for p in process_list if p.no_progress_termination) / n,
            per_task_outcome=outcome_list,
            per_task_process=process_list
        )

    def _compute_outcome_metrics(self, task: TaskSample, ep: EpisodeResult) -> OutcomeMetrics:
        """计算单任务的 outcome metrics"""
        cited = set(ep.cited_chunks)
        gold = set(task.gold_chunks)

        # Citation metrics
        intersection = cited & gold
        recall = len(intersection) / len(gold) if gold else 0
        precision = len(intersection) / len(cited) if cited else 0

        # Rubric completion
        rubric_completion = 0.0
        if task.rubric and ep.final_answer:
            answer_lower = ep.final_answer.lower()
            covered = sum(1 for kw in task.rubric.required_keywords if kw.lower() in answer_lower)
            rubric_completion = covered / len(task.rubric.required_keywords) if task.rubric.required_keywords else 0

        # Answer quality（简化：基于 rubric 和 citation）
        answer_quality = 0.5 * rubric_completion + 0.5 * precision

        return OutcomeMetrics(
            task_id=task.task_id,
            task_success=ep.termination == TerminationType.ANSWER and ep.final_answer is not None,
            answer_quality=answer_quality,
            rubric_completion=rubric_completion,
            citation_recall=recall,
            citation_precision=precision
        )

    def _compute_process_metrics(self, ep: EpisodeResult) -> ProcessMetrics:
        """计算单任务的 process metrics"""
        from collections import Counter
        from data.schemas import ToolName

        # TrajectoryStep.action is a dict with "tool" key
        tool_counts = Counter(ToolName(s.action["tool"]) for s in ep.trajectory)

        total_actions = len(ep.trajectory)
        invalid_rate = 0.0  # 需要从 trajectory info 获取
        repeated_query = 0
        repeated_read = 0

        return ProcessMetrics(
            task_id=ep.task_id,
            total_steps=total_actions,
            search_count=tool_counts.get(ToolName.SEARCH, 0),
            read_count=tool_counts.get(ToolName.READ, 0),
            rerank_count=tool_counts.get(ToolName.RERANK, 0),
            cite_count=tool_counts.get(ToolName.CITE, 0),
            invalid_action_rate=invalid_rate,
            repeated_query_rate=repeated_query / max(total_actions, 1),
            repeated_read_rate=repeated_read / max(total_actions, 1),
            mean_new_evidence_per_read=0.0,  # 需要更详细的 tracking
            no_progress_termination=ep.termination == TerminationType.NO_PROGRESS,
            termination=ep.termination
        )

    def generate_report(self, metrics: AggregatedMetrics) -> str:
        """生成评测报告"""
        lines = [
            "=" * 60,
            "ResearchAgent-RL Evaluation Report",
            "=" * 60,
            "",
            "### Outcome Metrics ###",
            f"Task Success Rate:     {metrics.task_success_rate:.2%}",
            f"Mean Answer Quality:   {metrics.mean_answer_quality:.3f}",
            f"Mean Rubric Completion:{metrics.mean_rubric_completion:.3f}",
            f"Mean Citation Recall:  {metrics.mean_citation_recall:.3f}",
            f"Mean Citation Precision:{metrics.mean_citation_precision:.3f}",
            f"Mean Citation F1:      {metrics.mean_citation_f1:.3f}",
            "",
            "### Process Metrics ###",
            f"Mean Steps:            {metrics.mean_steps:.1f}",
            f"Mean Search Count:     {metrics.mean_search_count:.1f}",
            f"Mean Read Count:       {metrics.mean_read_count:.1f}",
            f"Invalid Action Rate:   {metrics.invalid_action_rate:.2%}",
            f"Repeated Query Rate:   {metrics.repeated_query_rate:.2%}",
            f"Repeated Read Rate:    {metrics.repeated_read_rate:.2%}",
            f"No-Progress Term Rate: {metrics.no_progress_termination_rate:.2%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def generate_case_study(self, ep: EpisodeResult) -> CaseStudy:
        """生成单个 case study"""
        task = self.tasks.get(ep.task_id)
        outcome = self._compute_outcome_metrics(task, ep) if task else None
        process = self._compute_process_metrics(ep)

        return CaseStudy(
            task_id=ep.task_id,
            task_type=task.task_type if task else TaskType.SURVEY,
            user_query=task.user_query if task else "",
            trajectory_summary=self._summarize_trajectory(ep),
            key_decisions=self._extract_key_decisions(ep),
            evidence_used=list(set(ep.cited_chunks)),
            final_answer_preview=ep.final_answer[:200] if ep.final_answer else "",
            termination_reason=ep.termination.value,
            outcome_metrics=outcome,
            process_metrics=process,
            strength="TBD",
            weakness="TBD"
        )

    def _summarize_trajectory(self, ep: EpisodeResult) -> str:
        return " -> ".join(s.action["tool"] for s in ep.trajectory)

    def _extract_key_decisions(self, ep: EpisodeResult) -> List[str]:
        decisions = []
        for s in ep.trajectory:
            decisions.append(f"{s.action['tool']}: {s.action['intent']}")
        return decisions
