# Effect-Space Agent RL：新颖性复核与判别实验（2026-09-29）

结论先行：**当前的简单版本不够新颖，不应作为 A 会主张。**“按检索结果聚类、复用后续轨迹、对类内/类间分别加 KL”分别有很近的先例或属于已有数学恒等式。我们保留的只是一个**尚待验证的研究假设**：在真实多跳工具环境中，能否从工具实际执行结果构造可测、低误差的动作商空间，并据此给出可计算的、非空泛的策略改进保证，同时在同等生成 token/工具预算下优于已有方法。代码现在只准备了这个假设的前置反证实验，没有新优化器、训练结果或理论证明。

## 三轮近邻检索的碰撞点

检索范围包括抽象 MDP/策略梯度与 importance sampling、LLM trust region 与层级策略优化、Agent RL 检索树/状态聚类/过程优势。以下是最需要在论文 related work 中正面对比的原文；arXiv 预印本不等同于已中会论文。

| 候选主张 | 最接近工作 | 复核结论 |
|---|---|---|
| “首个抽象动作上的策略梯度” | [JMLR 2024: Policy Gradient Methods in the Presence of Symmetries and State Abstractions](https://jmlr.org/papers/v25/23-1415.html) | 已有 MDP 同态下的抽象策略梯度；不能宣称首创。 |
| “用相同/相似检索结果去重 Agent rollout” | [TreePS-RAG](https://arxiv.org/abs/2601.06922) | 已以检索集合相似度做树枝剪枝和过程优势；单纯检索聚类不够。 |
| “纠正 token ratio trust-region 失真” | [Rethinking the Trust Region in LLM RL / DPPO](https://arxiv.org/abs/2602.04879) | 已从直接策略散度入手；我们的卖点不能只是从 token ratio 改为 divergence。 |
| “类间/类内双层 gate” | [Fibration Policy Optimization / FiberPO](https://arxiv.org/abs/2603.08239) | 已提出多层级 base/fiber gating；要说明工具执行诱导的动作类为何有不同数学/计算性质。 |
| “Agent 状态近似等价并共享 credit” | [BiPACE](https://arxiv.org/abs/2606.25556) | 已用 actor hidden state 近似行为相似性，给出状态聚类偏差与动作条件化 credit。 |
| “把等价状态合并并给出近似误差论证” | [GraphPO](https://arxiv.org/abs/2606.18954) | 已有语义状态合并、共享后缀和在类稳定性假设下的偏差分析；仅加一个 residual 界仍不够。 |

KL 链式分解

\[
D_{\rm KL}(\pi'\|\pi)=D_{\rm KL}(\bar\pi'\|\bar\pi)
+\mathbb E_{z\sim\bar\pi'}D_{\rm KL}(\pi'(\cdot\mid z)\|\pi(\cdot\mid z))
\]

本身是概率恒等式。固定状态的类内价值差 \(\Delta_Q\,\mathrm{TV}\) 界也只是标准有界函数不等式，不构成新定理。把它们换上 Agent RL 符号不会产生新颖性。

## 值得继续验证、但仍不保证新颖的窄切口

1. **动作类由可重复的工具执行诱导**，而非仅靠文本嵌入、隐藏状态或采样轨迹层级。映射随真实状态变化：\(z=f_s(a)\)。同一有序检索 ID 仍非完整后继状态，因为历史里保留原 query，环境缓存不同分数供 RERANK 使用。
2. **先测近似同态残差，再决定是否共享 credit。** 目标不是凭相同 ID 声称等价，而是估计同类动作的继续价值差与转移差，并在高残差处回退到原始轨迹估计。现有脚本只能估计描述性的有限样本回报差，远未达到高置信证书。
3. **策略质量而非聚类本身作为目标。** 理论上需要处理类概率质量的可估性、有限候选采样误差、类内重排、状态占用漂移、off-policy 行为策略，以及工具随机性/版本变更。最终要有一个相对标准 GRPO 的可计算、非空泛改进下界或受控偏差结果。
4. **在相同总生成 token 和工具调用预算下比较**，且要正面对比 TreePS-RAG 式剪枝、GraphPO/BiPACE 式 credit 以及 DPPO/FiberPO 式稳定性。若这些不能形成可验证的差异，应停止“效果空间 trust region”叙事。

这不是“我们已经有了新算法”的声明；新颖性仍要随着 2026 年后续论文持续复核。特别是 [GraphPO 附录](https://arxiv.org/html/2606.18954)已有近似等价偏差分析，[BiPACE 全文](https://arxiv.org/html/2606.25556)已有 Agent 行为分组的理论与实证，不能忽略。

## 明天先做的反证实验

使用 MuSiQue **train** 的同一任务前缀，采样多个首轮 SEARCH，真实调用 split-wide 检索器。第一关是**不同 query、相同 topk、相同有序返回 ID** 的碰撞率，而不是相同文字重复率。若它低到难以找到配对，当前精确商空间没有足够覆盖率。第二关对每条 query 独立采样后续轨迹，比较同类/异类回报差；记录相同 ID 下分数差，并以同一 query 重复采样的 split-half 差作噪声尺度。`original` 记录真实原 query 历史，`canonical` 仅把模型看到的首轮 assistant 文本替换为类代表，环境仍执行原 query。因此 canonical 是**诊断性干预**，不是现有环境的训练/评测分布。

预设停机条件：无足够不同 query 的精确碰撞，或同类原始历史的回报差不低于异类差/采样噪声，或分数差与 RERANK 导致系统性价值差，都不应贸然共享 continuation。门槛应在读结果前固定；不能事后调阈值制造结论。10 个配对任务只是链路 smoke，不是统计结论；至少扩大至多题、多次 continuation，按**题目**而非轨迹 bootstrap，并保留失败样本。

当前实现入口：[首轮采样与召回诊断](../scripts/effect_space_pilot.py)、[配对 continuation](../scripts/effect_space_continuation_pilot.py)、[详细研究计划](effect_space_research_plan_cn.md)、[服务器运行说明](effect_space_server_runbook_cn.md)。
