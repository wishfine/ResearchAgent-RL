# Effect-Space Policy Optimization：算法假设与预注册实验计划

状态：**阶段 0 可行性诊断已实现；尚未接入 Vime 在线训练，也没有模型采样结果。** 不应把本文件当作算法有效性的证据。

## 研究问题

在同一 Agent 前缀状态 (s)，语言模型可能生成多个不同的 SEARCH query (q_i)，而固定检索器返回完全相同的有序 chunk 序列 (z_i=R(q_i))。普通轨迹式 GRPO 仍分别支付后续生成成本。如果这个“语言动作到工具效果”的多对一映射普遍存在，能否在**不改变目标策略梯度，或明确控制偏差**的前提下，按效果分配后续 rollout 预算与信用？

当前代码中“精确观察签名”定义为 SEARCH observation 中**有序**的 chunk ID 元组。无序集合或 Jaccard 高重叠仅作诊断，不能称为等价。即使签名相同，它也还**不是**完整 MDP 状态等价：后续对话保留原 query，环境还保留检索分数，而 RERANK 会使用这些分数。代码见 `research_agent/core/effect_space.py`；200 题、每题 8 次第一步采样脚本见 `scripts/effect_space_pilot.py`。

最强的相关工作不是普通 GRPO，而是 [TreePS-RAG](https://arxiv.org/html/2601.06922v1)：它在共享前缀树中，以 top-k 检索集合的 Jaccard 相似度聚类剪枝，并从后代终局奖励估计步骤优势。它在 NQ/HotpotQA 训练，对包括 2Wiki/MuSiQue 在内的七个 QA 集评测；对照 Search-R1、去掉过程优势和去掉相似度剪枝的消融，组大小为 8，并保持近似的 rollout 预算。因此“按证据相似度去重”**不是我们的创新点**。[TIPS（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/file/fd881d3b625437354d4421818f81058f-Paper-Conference.pdf)也已经给搜索轮次设计势函数奖励，报告 EM/F1、held-out 学习曲线和消融。我们的差异必须落在**效果类上的概率/价值估计与误差分析**，不能只是另一种剪枝阈值。

## 暂定算法，而非既成结果

对同一前缀采样 (N) 个 query，调用固定检索器，按有序结果 (z) 分组。对于每个独特效果，采样若干下游 continuation，得到 \(\hat Q(s,z)\)。将其映射回每个原始 query 的 score-function 项：

\[
\hat g=\frac1N\sum_{i=1}^N\nabla_\theta\log\pi_\theta(q_i\mid s)
\left(\hat Q(s,z_i)-b(s)\right).
\]

这里 (b(s)) 必须只依赖共享前缀，不能偷偷依赖当前动作。如果所有同效果 query 的**真实后续状态和后续策略分布相同**，且 continuation 回报估计无偏，则共享价值可用于上述估计器；这只是待严格证明的命题，不是当前代码已保证的性质。

**重大障碍：** 项目现有 ChatML 轨迹保留了模型生成的 SEARCH action，后续模型能看到原 query；候选检索分数也留在环境内部，可能改变 RERANK。即使检索到同一批 chunk，真实后续状态也不同。因此直接在 Vime 里复用同一 continuation 会引入状态抽象偏差；将 query 从后续 prompt 删去、或丢弃分数，又会改变训练/评测环境，且需要重新设计多轮 token/log-prob 对齐。阶段 0 不修改现有训练路径。阶段 1 必须先做“同签名、不同 query”的 continuation 对照，决定走**经验证的近似抽象**、真正的规范化环境，还是放弃这个方向。

## 阶段 0：无训练的可行性 pilot

数据只用 MuSiQue train：从 19,938 题中按固定种子取 200 题，按 2/3/4 hop 分层报告；每题在相同初始前缀下独立采样 8、16 个 SEARCH 动作。另取真实轨迹中的第二跳/第三跳前缀复测，避免“第一步碰撞”外推至多跳过程。模型采样数据和检索分析分开保存为 `samples.jsonl`、`effect_records.jsonl`、`summary.json`，便于离线重算。

主要观测：action 解析率、SEARCH 比例、不同 query 数、不同有序效果数、不同 query 却同效果的比例、效果类大小、无序相同但排序不同的比例、近似效果的 Jaccard 分布、各类 gold-hop 命中率。所谓“每类只运行一次 continuation 的节省”先只报**假想 continuation 数**，不得写成实际 GPU 加速。

预注册继续条件：至少约 20% 的不同 query-action 落在已有精确效果类，且不是同一 query 文本的简单重复；第二/第三跳也有可观测冗余。若明显低于此门槛，优先研究有偏的近似效果类是否仍有足够低的类内价值差异；若也不成立，停止本方向。20% 是 pilot 决策阈值，不是论文结果。

示例（在有 OpenAI-compatible 模型服务的服务器执行；**不会启动训练**）：

```bash
cd ~/ResearchAgent-RL
BASE="/data/$USER/research-agent-rl-data"
OUT="$BASE/outputs/effect_space_pilot_train200_n8"
mkdir -p "$OUT"
PYTHONUNBUFFERED=1 python scripts/effect_space_pilot.py \
  --tasks_dir "$BASE/musique_rl_v2/tasks/train" \
  --corpus_dir "$BASE/musique_rl_v2/corpus/train" \
  --model_url http://127.0.0.1:8001/v1 \
  --model_name Qwen3.5-9B \
  --max_tasks 200 --samples_per_task 8 \
  --selection_seed 42 --temperature 1.0 --top_p 0.95 \
  --output_dir "$OUT"
```

模型服务、权重及端口必须按实际情况设置。服务器采样随机性可能不完全由 Python 的选题 seed 控制，所以**必须保存 `samples.jsonl`**；重算检索指标时使用 `--samples_file "$OUT/samples.jsonl"`，而不要重新抽样。采样输出是研究原始记录，先不要提交到 GitHub。
脚本逐条落盘；若连接中断，可在**完全相同参数**后加 `--resume` 续采，参数不匹配会拒绝混合样本。

## 阶段 1：验证状态抽象与估计器

从 pilot 中选有至少两个不同 query 的效果类。对每个 query 独立运行多次剩余轨迹，测量同效果类内与类间的回答正确率、证据链完成率、奖励均值/方差及下一步动作分布；分别在原始历史与候选规范化历史下测。若原始历史的类内差异显著，不能使用“精确等价”表述。需与独立 continuation 的 Monte Carlo 估计、TreePS 风格 Jaccard 聚类剪枝以及只缓存重复检索的工程基线对比。所有方法统计**总模型生成 token、SEARCH 调用、完整 continuation、墙钟时长**，不能仅固定轨迹条数。

## 阶段 2：正式 RL 与论文评测（仅在阶段 1 通过后）

1. 从 MuSiQue train 划出固定的内部 validation questions；只用剩余 train 训练和调参，官方 dev 2,417 题只在模型选择锁定后评测。train/eval 语料依现有 split 隔离。
   在任何方法对比前，还须固定 READ 返回长度：现有实现只返回 chunk 前 300 个字符，多跳答案常在后文。若调整阅读窗口，**所有**基线都要在相同新环境中重跑；不能把观察能力提升算作效果空间算法的收益。
2. 同一初始化、同一检索器、同一工具协议、相同 (N=8)、相同奖励和 KL：比较 SFT-only、标准 GRPO、仅检索结果去重、TreePS 风格相似度剪枝、完整效果空间方法。若完整 TreePS 复现成本过高，要明确写成近似基线，不能冒充原论文实现。
3. 预注册消融：有序精确效果 vs 无序集合 vs Jaccard 近似；是否共享价值；固定 vs 按不确定性分配 continuation；保留 vs 规范化原 query 历史；(N=4/8/16)。
4. 主指标：MuSiQue answer EM/F1、全部支持证据召回和严格 task success；辅助指标：分 hop 成功率、解析成功率、非法动作率、平均步骤、重复检索、有效效果多样性。画相同生成-token/工具调用预算下的质量曲线，而不只画相同步数的质量曲线。
   同时记录训练期 group reward 方差、非零 advantage 比率与 policy-gradient loss；旧 HotpotQA GRPO 曾出现奖励饱和，不能仅以 job 成功和 reward 均值上升判断算法有效。
5. 至少三个随机种子；checkpoint 先由内部 validation 固定选择，再报告 dev 结果、配对置信区间与失败类型。额外在 2Wiki/HotpotQA 的全库设置上做跨数据集验证，避免只证明适配 MuSiQue 的 FTS5 实现。

计划图：Figure 1 为不同采样数下的 query 数、独特效果数与类大小分布；Figure 2 为 matched-budget 的成功率/证据链完成率 Pareto 曲线。没有模型 pilot 或 RL 结果之前不生成假图。

## 当前实现边界

`effect_space.py` 只提供精确分组与“在额外假设下”的优势映射；pilot 只采第一步并做检索诊断。**没有**接入 Vime rollout group、没有状态规范化、没有证明无偏、没有训练出新模型。下一项工程工作取决于阶段 0/1 的真实测量，不能跳过这个 go/no-go 决策。
