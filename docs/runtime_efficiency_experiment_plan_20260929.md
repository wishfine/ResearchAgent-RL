# Agent 运行时效率：现象验证与干预实验计划（Luna 交接版）

日期：2026-09-29。状态：**规划完成，实验代码待实现，服务器实验未运行**。

## 0. 给接手者的任务说明

本轮任务不是实现一个已经确定的新 RL 算法，而是建立可信的诊断实验，回答：

> 在同样的题目、模型和检索协议下，是否存在更便宜且成功率不明显下降的执行方式？当前策略为什么没有采用它？

依次完成：离线审计 → 可恢复续跑基础设施 → 小规模冻结策略采样 → 前缀条件下的动作对照 → 分析与 go/no-go 报告。**不要直接启动 SFT/GRPO，不要宣称新颖性成立，不要预填预期提升数值。**

用户后续会让 Luna 实现和运行；本文件中的新脚本名、CLI 和 schema 是待实现契约，并非现有可运行功能。首次交接时先检查工作区和本文列出的源码。不要覆盖已有未提交工作，不要主动提交或推送，不要替用户停掉其他 GPU 任务。

### 首轮完成标准

- CPU 测试验证：前缀恢复正确、分支隔离、无未来信息泄漏、预算/断点续跑正确。
- 至少完成 P0 离线审计和 P1 小规模端到端 smoke。
- P1 通过后才能运行 P2；给出数据证据决定是否进入训练研究。
- 结果与基础设施日志分开；结果必须可追溯到原始样本、模型、数据、代码和协议版本。
- “没有观察到可压缩空间”也是合法结论，不能为得到显著改善而不断改选样本。

## 1. 已知事实与不能沿用的假设

2026-09-29 对本机文件做了只读统计：

| 文件 | 样本/唯一 task 数 | 步数分布 | 完整四步动作序列 |
|---|---:|---|---:|
| `artifacts/e07_sft874_full/trajectories.jsonl` | 3000/3000 | 4 步 2985；5 步 1；6 步 14 | SEARCH→READ→CITE→ANSWER：2984 |
| `artifacts/e08_grpo_v2/eval_n100/iter_0000099/trajectories.jsonl` | 100/100 | 4 步 100 | SEARCH→READ→CITE→ANSWER：100 |

这些是实际读取文件的统计，不是新算法效果。P0 必须重新生成并校验，以文件 SHA256 标识输入。

旧 HotpotQA 设置下四步是流程规定下的基本路径，不能拿它证明“有大量冗余动作”。每题一次记录也不足以比较同一策略不同路径的成功概率。不同 checkpoint 的输出不得混为同一策略的独立重复样本。

当前 MuSiQue 数据见 `data/musique_rl_v2/stats.json`：

- 官方 train 来源 19,938 题；官方 dev 来源 2,417 题；不是官方有标签 test。
- train 语料 83,866 chunks，eval 语料 21,100 chunks。
- `retrieval_scope=split_corpus`，`reference_docs=[]`，模型可以多轮检索。
- 2/3/4-hop 数量不均衡。旧四步 SFT 模型能否适应新协议，需要先测，不能默认成功率较高。

以下说法只能作为待检验假设，不能写进结果结论：

- 高熵就冗余，低熵就有捷径。
- 更短一定更好，重复搜索一定没有用。
- 当前策略依赖某动作，就意味着所有策略都必须依赖它。
- 固定 baseline 不合法，只有 V(s) 能抑制绕路。
- 训练梯度方差下降，自动意味着部署 token/tool 成本下降。

## 2. 实验问题与可证伪假设

| 编号 | 假设 | 需要观察什么 | 什么会否定或削弱它 |
|---|---|---|---|
| H1 | 同题同策略存在成功但成本不同的路径 | 每题多次独立采样后，成功样本成本有差异 | 大多固定流程，或所有低成本轨迹都失败 |
| H2 | 存在前缀可生成的便宜替代动作 | 不读未来、不读 gold 的替代方案经独立续跑验证 | 只有看过答案或未来 query 才能造出短路径 |
| H3 | 一些“看似重复”动作没有足够可靠性收益 | 对照动作/替代动作的成功差和成本差 | 去掉信息获取机会后明显掉成功率 |
| H4 | 失败不是单纯检索覆盖率不足 | 诊断检索能找到证据，但策略花费更多或选择失误 | 连可用检索的 evidence coverage 都很低 |
| H5 | 存在对当前策略的可学习改进空间 | 独立验证的替代动作改善成本—质量关系 | 所有改善都只是筛选噪声，或不符合部署可见信息约束 |

H1 的“最便宜已观察成功样本”只是一项描述统计，不是真正最短路径，也不是该路径可靠性的估计。

首轮不检验“成本惩罚导致训练坍缩”的因果命题：那需要后续受控训练实验，不可能从冻结模型的路径长度统计直接得出。

## 3. 数据、模型与协议锁定

### 3.1 数据分组

1. 从 MuSiQue **train** 选择题目，不用官方 dev 来反复调方法。
2. 固定 `selection_seed=20260929`；先形成 manifest，再发任何模型请求。
3. `smoke_tasks`：24 题，每种 hop 8 题；`discovery_tasks`：96 题，每种 hop 32 题，包含 smoke 子集。
4. `confirmation_tasks`：另外预留 96 题，与 discovery task ID 不重叠；首轮不自动运行。
5. hop 来自离线元数据，仅用于抽样分层，不把 gold decomposition/中间答案放入提示词。
6. 分层均衡样本的宏平均不是自然数据分布平均。两种都可以报告，但明确权重；报告各 hop 结果。
7. 本轮 train 内 task 隔离不等于所有实体/证据内容完全隔离；另报重复内容哈希，避免过强泛化声明。
8. 若后续训练，诊断与 confirmation 题目留出，不加入训练；最终官方 dev 只在协议冻结后评估，称 held-out dev evaluation。

### 3.2 模型

- 首选已有 Qwen3.5-9B SFT checkpoint，**实际路径由服务器盘点确认**；记录 config/tokenizer/chat-template 和权重索引指纹。
- 原始 Qwen3.5-9B 作为可选迁移控制，先用同一 24 题协议测试；不是首轮必须同时部署两个模型。
- 旧历史模型路径 `/home/zhangyonglin/models/models/Qwen--Qwen3.5-9B/snapshots/master` 只能作查找线索，不能假定仍存在或代表 SFT。
- 不把旧 HotpotQA 分数和新 MuSiQue 分数直接比较为模型提升。
- 所有主体实验冻结权重，模型端点不得中途换权重。manifest 核验失败要停止。

### 3.3 默认采样设置（在 P1 后锁定）

| 参数 | 初始值/规则 |
|---|---|
| policy prompt | 现有 `MULTIHOP_SYSTEM_PROMPT`，保存全文和 hash |
| temperature / top_p | 0.7 / 0.95 |
| max_steps | 15，每个有效/无效动作尝试都按统一环境规则计数 |
| max output tokens / turn | 512 |
| max_invalid_actions / max_no_progress | 先审计当前实现；如修复，升协议版本，所有新实验统一使用 |
| context limit | 从实际部署配置读取并锁定；上下文超限单独记录，不静默删历史 |
| tool configuration | 所有组相同，SEARCH/READ/RERANK/CITE/ANSWER 保持原接口 |
| tool results | 主实验用固定本地 corpus，固定检索/READ 实现，避免实时网络变动 |
| request seed | 稳定派生并传到服务端；服务不支持时明确记录，不假装可精确复现 |

不能把 `selection_seed` 当作模型采样 seed。当前 `LLMClient.generate_response()` 没有 seed 参数，需向后兼容地补充并测试。相同服务端 seed 也不保证动态 batching 下 bitwise 一致，不能宣称由它实现了无偏的 common-random-number coupling。

## 4. 先审计现有实现，再决定复用边界

| 现有位置 | 可复用部分 | 必须检查/补充 |
|---|---|---|
| `research_agent/core/baseline_runner.py` | episode 记录、旧指标 | `run_episode()` 总是 reset；需抽取兼容的 resume runner；旧 task_success 非严格 EM |
| `research_agent/core/env/env.py` | 注册工具、执行、终止 | 快照/恢复、失败分类、确定性回放 |
| `research_agent/core/env/state.py` | candidates/reads/citations/history/预算 | `check_no_progress()` 当前只比较列表长度；等数量替换/排序不能被误判；完整保存私有计数 |
| `research_agent/core/env/rollout_collector.py` | 多跳提示词、历史渲染 | 保存实际 segments 和实际发给模型的 messages；不要把简短 Observation 当成全部状态 |
| `research_agent/core/env/llm_client.py` | API 与 usage | seed、finish_reason、请求/工具耗时、实际 payload hash、缺失 usage 标志 |
| `scripts/run_llm_baseline.py` | load_task/make_env/readiness | 旧 CLI 不是多样本/可续跑实验驱动；别套循环覆盖同一输出 |
| `scripts/effect_space_continuation_pilot.py` | manifest/hash/独立续跑分析模式 | 仅第一 SEARCH 的同检索结果对照，不是任意动作删除实现 |
| `scripts/decision_value_pilot.py` | 部分离线指标组织 | answer entropy 不等于动作价值；不能把该实验变成在线奖励 |

具体注意：`check_repeated_action()` 目前按连续同 tool 类型累计；连续两次 READ 或 SEARCH 不一定重复。不要把该计数命名为“冗余动作数”。

P0/P1 若发现 progress/终止/工具错误处理的真实 bug，用 focused regression test 修复。禁止关闭终止保护来掩盖 bug；也禁止悄悄重算旧历史结果并覆盖旧台账。留存旧记录，记录新 protocol 和 diff。

## 5. 快照与分支语义：这是最重要的实现部分

### 5.1 什么是决策前缀

在模型生成当前动作 **之前** 定义 `prefix_id`。快照包括：

- 任务 ID、数据/语料版本；gold 字段只留在评测侧，actor/proposer 只能接收 public view。
- 完整模型可见历史：system、user、之前的 assistant 动作及工具 observation，原始与实际 API 渲染均可追溯。
- 候选 ID/顺序/分数、已读 summaries、引用、搜索/失败历史、轨迹、final/done 状态。
- current/remaining/max steps、invalid/no_progress/repeated counters 和 `_prev_*`。
- 工具注册与参数、随机数状态或请求种子序列、prefix 成本账本。
- 语料仅以固定只读 handle/路径+hash 共享；不可 deepcopy 活跃 SQLite connection。可变 state/collector 必须逐分支隔离。

现有 `steps[t].observation` 并不等于以上快照。旧日志可用于 P0；若要回放旧日志，须重新执行前缀并核验每步结果和 prompt hash，一旦不一致就拒绝用于干预。

### 5.2 三种实验分支

所有分支来自同一合法非终止前缀 h；参考动作 a 是 discovery 中已采样的动作。主实验仅针对 SEARCH/READ，暂不删 CITE/ANSWER，不改变工具协议的最低必要开销。

**KEEP_A：** 恢复 h，执行参考动作 a，把该动作及真实结果加入历史，然后用冻结策略重新生成全部后续。

**REPLAN：** 恢复 h，不执行 a，不插入 a 的文本或 observation，直接让原策略重新采样当前动作及后续。允许再次选到 a，并报告再次选中率。

**ALT_PREFIX_ONLY：** 恢复 h，执行一个仅根据 h 生成、经当前 schema/可见 ID 检查的替代动作 a'，然后重新生成全部后续。仅接受合法 SEARCH/READ；不允许用未来实体、最终证据或 gold 拼 query。

必须正确解释：

- REPLAN 比较的是固定动作的 Q(h,a) 与原策略重采样的 V(h)，**不是纯动作删除实验**。
- 没有状态变化、没有额外提示的“跳过后续跑”就是重新规划；不能把它包装成确定性 bypass。
- 若禁止再次生成 a、修改 prompt 让模型走捷径，已经改变干预策略；必须是另一个命名的探索组，不能混进 REPLAN。
- ALT 验证的是特定可执行动作替换，不是该动作在所有策略下“不必要”的证明。
- 原动作可能是有信息的桥接搜索，即使最终不引用它也不能删后保留未来 query。
- 所有分支都重新生成 suffix，不能照抄原来的后续动作/答案。

### 5.3 替代动作的生成与选择

1. proposer 使用同一个冻结模型和 h 的 public view；可加明确的“给出另一种合法检索/阅读选择”提示，保存模板 hash。
2. 每个前缀最多生成 4 个候选动作；禁止提供 suffix、奖励、成功标签、gold ID 或 oracle decomposition。
3. READ 仅选择当时已搜索到的 ID。SEARCH 参数合法，不自动补入正确实体。
4. 去掉与 a 完全相同的规范化动作；不把 embedding 相近直接当作因果等价。
5. 初版按预注册顺序取首个合法不同动作，不用验证集回报挑赢家。不强求当下动作更便宜，最终比较剩余总成本。
6. 无合法候选标记 `no_eligible_alternative`，保留统计；不悄悄换成别的“容易改善”前缀。
7. 如果后续用少量续跑挑候选，必须新增 screening split；screening 的全部成本入账，最终 validation 用全新 samples。

proposer 开销属于实验获取/选择成本；若未来部署时也调用 proposer，则必须计入部署成本。当前强制替代续跑的改善不能冒充已有自主策略改善。

### 5.4 步数与费用的公平性

各组从相同 `remaining_steps` 起步，不补偿 KEEP_A 已执行的一步，不给任何组无限续跑。KEEP_A 执行 a 消耗一步，REPLAN 执行重新采样的动作也消耗一步。

分别记录：

- `prefix_sunk_cost`：各组相同的既有历史成本。
- `suffix_policy_cost`：从决策边界起的实际策略/工具成本。
- `forced_action_cost_imputed`：KEEP/ALT 强制动作未在现场重新解码，部署成本估计中应计入一次同前缀生成及该动作输出；标明估算，不能把强制动作的模型计算视为免费。
- `experiment_actual_cost`：真实 proposer、screening、所有分支、重试、失效请求消耗；绝不混同部署推断成本。

不把 prompt_tokens 当 GPU FLOPs。缓存 token 若服务端未给出则置 null/unknown，而非 0。报告输出 token、累计输入 token、工具调用和 wall time 四项原始口径；成本加权结果仅作透明的补充。

## 6. 分阶段运行与停止条件

### P0：仅离线审计，0 模型调用

- 重现第 1 节两份文件统计；扩展到已有 base/GRPO 文件，但分模型分协议报告。
- 统计 actions 数、每种 tool 次数、token、成功/失败、终止原因、同题样本数和缺字段。
- 输出长度—结果相关性时显式写 `descriptive_not_causal=true`。
- 检查新 MuSiQue task scope、语料、提示词；新增测试确定 public prompt 中不含 gold。
- 输出 `offline_audit.json`、`offline_audit.md` 和协议差异表。

P0 不决定新算法优劣，只回答数据可用性和旧设置的局限。

### P1：CPU 测试 + 服务器 smoke

1. 所有 snapshot/leakage/resume 测试通过。
2. 24 题 × 4 次独立完整 rollout = **96 episodes**，保存成功和失败。
3. 从上述题中预注册规则选最多 6 个合法前缀，每题最多 1 个；每前缀 3 arms × 2 次 = **最多 36 条续跑**。
4. 主干和分支命令可中断恢复，清单和模型版本核验有效。

通过条件：无 gold/未来信息泄漏、回放一致、没有跨分支状态污染、指标/成本完整；基础设施错误低于预注册 5% 门槛，否则先修基础设施，不解释模型优劣。

如果格式失败率 >20% 或几乎无法得到合法 SEARCH/READ，先归因为协议适应性问题；允许单独比较 base 与 SFT 的 24 题 smoke，但不直接继续大量干预，也不默认开始训练。

### P2：主体诊断，仍不训练

- discovery：96 题 × 8 次 = **768 episodes 总量**；若 P1 协议相同可复用已完成 sample ID，不能重复计算样本。
- 从符合条件的 discovery 中选择最多 24 个前缀，尽量每 hop 8 个，每题最多一个。
- 每前缀 KEEP_A/REPLAN/ALT 各 8 个新续跑 = **最多 576 条续跑**。
- 候选动作生成额外最多 24×4 = 96 次单轮请求，不算完整 episodes。
- 8 个续跑不足以对单个前缀证明 2 个百分点非劣；初版主要报告跨任务聚合差异与区间，并标记单前缀结论不确定。

前缀选取：按 task seed 排序，在合法 SEARCH/READ 决策点中选一处；可以预注册分层（例如已检索结果重复程度），但**不按最终成功或改善回报筛选**。候选是否可用只检查 schema/public-state。

### P3：独立确认，须先交付 P2 报告

- 若 P2 有值得继续的机制信号，在预留 confirmation 题目重复完整协议。
- 依据 P2 的 task-level 方差与目标精度做样本量估算，而非机械认定 96 题足够。
- 固定脚本/提示词/选择规则；更多重复只能用新 sample IDs，不用“跑到显著就停”。
- 在第二模型或第二全库多跳数据设置复核后，再考虑普遍性。旧 task-scoped HotpotQA 只作协议控制，不当作强独立复现。

### Go / No-Go 决策

| 结果 | 下一步 |
|---|---|
| 绝大部分为协议下最低步数，token 也无明显差异 | 停止“动作压缩”主线，考虑观察长度或检索质量；不得继续宣称绕路问题 |
| 同题有短成功样本，但新续跑无法复现可靠性 | 这是选样幸运，不作为偏好监督可靠标签 |
| 前缀合法替代在独立续跑中节省成本且不明显损伤质量 | 开始研究如何让原策略学到这些选择；先做简单基线 |
| 替代普遍降低可靠性 | 额外动作可能有实际价值；不能强制写成“虚假权衡” |
| 只有 oracle/未来信息辅助才有效 | 只能作上界诊断，不能作在线方法效果 |
| CI 太宽 | inconclusive；报告需要的样本量和预算，不冒充“无差异” |

建议用于规划的非劣容忍度：grounded success 下降不超过 **2 个百分点**，成本相对减少目标 **10%**。二者是预注册决策阈值，不是预测结果，也不保证当前 pilot 有足够统计功效。

## 7. 指标与统计口径

### 7.1 成功和质量

保存旧 `task_success` 但命名 `legacy_task_success`。当前 `evaluate_episode()` 使用 answer_quality≥0.5 和 citation recall 阈值，它不是严格答案正确率。

新诊断主要报告：

- 版本化的 answer normalized EM、token F1，支持 aliases。若声称 official MuSiQue score，必须使用并注明官方 evaluator 版本；否则标注 custom EM/F1。
- `grounded_success = answer_EM && all_gold_supports_cited && all_cited_ids_read_and_in_scope`，阈值/布尔定义在 protocol 固定。
- citation precision/recall/F1，answer_submitted rate，parse success，invalid rate。
- 支持段落并非永远是唯一可接受证据，严格 citation 指标可能误伤替代证据；失败案例做人工抽检，不用临时放宽规则抬分。

每组同时报告全样本成本和成功样本条件成本。只报告成功样本成本容易掩盖“失败更快”的策略。

### 7.2 估计量

对每个前缀 i、arm a，用独立续跑均值估计成功率和 suffix 总成本：

`delta_success_i = mean(Y_ALT,i) - mean(Y_KEEP,i)`

`delta_cost_i = mean(C_ALT,i) - mean(C_KEEP,i)`

REPLAN 单独对 KEEP 做同样比较。成功差以百分点报告；成本同时报绝对差与相对差。

- 先每题聚合，再 task-macro 平均，不让多前缀/多重试任务权重更大。
- 95% CI 用 task-cluster bootstrap（建议 2000 次，固定 analysis seed）；保留同题各 arms，不把 steps 当独立样本。
- task 数很少时说明 bootstrap 局限；单前缀可报 Wilson 区间，不以 8 次样本做强“安全删除”结论。
- 主检验预指定 ALT vs KEEP；REPLAN 为解释性对照，额外多重比较应校正或明确 exploratory。
- 非劣要求差值区间下界 > -0.02；CI 跨越门槛是“不确定”，不是成功。
- discovery 上挑出的最好路径/动作，不用同一批随机续跑验证；P3 必须有独立任务。

基础设施错误单列，不悄悄从分母删除。预注册重试次数（例如最多 2 次）；最终仍失败的 logical sample 保留。另报完成样本分析和把缺失视为失败的敏感性分析，任一组错误率高则暂停解释。

### 7.3 必须生成的图表

1. 每种 hop 的成功率、动作数、输入/输出 token 分布。
2. 同题 observed success-cost 分布（标题明确 descriptive）。
3. KEEP/REPLAN/ALT 的 task-level 成功差与成本差及 CI。
4. 对照前缀上原始重复程度与干预收益的关系，不画成因果定律。
5. 失败类型：未检索到证据、未读、缺引用、答案错、无提交、格式错、预算终止、基础设施错。

不需要在首轮制造训练曲线，因为没有训练；图上不得放未经测量的“20–40% 节省”。

## 8. Luna 的实施任务清单

建议目录（均为拟新增；先检查是否已有同名实现）：

| 顺序 | 交付物 | 文件建议 | 验收 |
|---|---|---|---|
| T1 | 离线审计 | `scripts/audit_runtime_efficiency.py` | 重现 P0 数字，失败/缺字段不崩溃 |
| T2 | 协议与 schema | `research_agent/diagnostics/runtime_efficiency/protocol.py` | 配置 hash、public view、样本唯一键 |
| T3 | 快照和续跑 | `research_agent/diagnostics/runtime_efficiency/snapshot.py`、`runner.py` | 运行前缀、恢复、继续一致；与旧 runner 兼容 |
| T4 | 多样本采集驱动 | `scripts/runtime_efficiency_pilot.py` | 子命令 prepare/collect/prepare-branches/continue/analyze |
| T5 | 独立统计 | `research_agent/diagnostics/runtime_efficiency/analysis.py` | 等任务权重、区间、成本与错误分类 |
| T6 | 服务器 runbook | `scripts/run_runtime_efficiency_pilot.sh` | 显式环境/路径、nohup -u、PID/exit status、预算和续跑 |
| T7 | 测试与报告 | `tests/test_runtime_efficiency_*.py`、结果报告 | CPU tests + P1 证据 + P2 go/no-go |

优先小而清楚的实现，不引入训练框架、不重构全仓库；如已有实验模块可安全复用，允许调整文件布局，但保持接口与语义。

### 必须覆盖的测试

1. JSON snapshot round-trip；corpus handle 不序列化，恢复校验 corpus hash。
2. 分支读到新 chunk 不改变其他分支；counter/list/collector 均独立。
3. KEEP 回放与原前缀执行得到同一工具结果、prompt、状态 digest（使用 deterministic fake actor）。
4. REPLAN 允许重复原动作；不会把原动作或未来 observation 放回 prompt。
5. 放一个只存在于 gold/suffix 的 sentinel，验证所有 actor/proposer API payload 不含它。
6. 桥接 query fixture：删前一步后不能保留未来实体，必须重新生成后续。
7. XOR/负证据 fixture：新 chunk 数为零或答案熵不变，不自动标记无用。
8. READ 未暴露 ID、引用未读 ID、跨 split ID 被拒绝。
9. no_progress 检查列表长度相同但内容/有效排序变化的情况；同 tool 连用不自动等于重复。
10. 合法工具失败、模型输出截断、HTTP timeout、context overflow 分开记录。
11. 流式写入中断恢复、不重复 sample、manifest 变化拒绝 resume、截断尾行显式恢复并保留审计。
12. seed 参数确实进入 API payload；无 usage 时记 unknown，不能合成 0 token。
13. 成本包含 failed attempts 和 proposer；强制动作部署估算与实际 API 消耗分开。
14. bootstrap 按 task 聚类；同题加倍续跑数量不能改变该题总体权重。
15. no_eligible_alternative 保留且不给伪造 advantage/成功指标。
16. 每批启动前检查剩余调用/token 预算，不能只在结束后发现超额。

## 9. 输出 schema、断点续跑和审计

每个 run 建议结构：

```text
run/
  protocol.json              # 全部参数、指标定义、指纹
  task_manifest.json         # 分层抽样、train/discovery/confirmation IDs
  environment.txt            # Python/packages/model server/code versions
  source.patch               # 涉及运行代码的未提交差异，检查无密钥
  notes.txt                  # 每次启动、修改、重试、停止原因
  records/discovery.jsonl
  records/prefix_manifest.jsonl
  records/proposals.jsonl
  records/continuations.jsonl
  snapshots/<prefix_id>.json
  reports/offline_audit.json
  reports/summary.json
  reports/report.md
  reports/figures/
  logs/driver.log
  logs/server.log            # 仅此实验拥有的服务
  pid.txt
  exit_code.txt
```

记录至少含：`protocol_version/run_id/task_id/hop/model_id/model_fingerprint/phase/sample_id/prefix_id/arm/request_seeds/status/termination_reason/messages/actions/tool_results/token_usage/timings/costs/metrics/error`。

prefix manifest 还含：`public_prefix_hash/state_hash/reference_action/candidate_selection_rule/selection_uses_labels=false/source_sample_id`。包含 ground truth 的评测记录要与模型输入路径隔离。

唯一键使用 `(protocol_hash, model_fingerprint, task_id, phase, prefix_id, arm, sample_index)`；不能只用 task_id。分支种子从唯一键和 seed block 稳定派生；不要复用相同 seed 来伪造独立重复。

写入实时 flush，单 writer 或加锁；失败留痕。汇总只用验证通过的 committed records，不把半行 JSON 当完整样本。完成由计划样本状态清单确认，不由 PID 消失或一句 “Done” 确认。

## 10. 资源预算与服务器运行约束

初版是单模型冻结推理：优先 **1 张确认空闲的 A800 80GB**，TP=1；是否能容纳模型与上下文由实际 smoke 验证。不需要默认占 4/6 张卡。

- GPU7 只是候选，不是当前已授权空闲卡；先用 nvidia-smi 和进程命令核验。禁止全局 `ray stop --force` / `pkill`。
- 旧训练环境和服务环境可能不同；先打印实际 Python、torch、vLLM、CUDA 库路径。不要改包/降级包来迁就未经验证的新脚本。
- 不假定服务器 SSH 已连接。用户未提供本轮可用通道时，交付完整自包含命令；不把本机路径当远端路径。
- 输出、缓存、TMPDIR 放 `/data/$USER/research-agent-rl-data/runtime_efficiency/`。项目 `results/` 已可能含跟踪文件/软链，不替换它；仅创建唯一命名子链接，路径冲突即停止。
- 每段命令都定义 PROJECT/BASE/RUN_DIR/MODEL_URL/MODEL_NAME，不依赖前一个终端的变量。
- 单独端口，先 `/v1/models` 核验精确 model id，随后执行一条有效动作请求；有超时且检查 server PID，不能无限 curl。
- `python -u`、nohup、每个 run PID、退出码、持续日志和 resume 都要提供。

P2 上限粗算：768 个完整 episodes + 576 个最长 15 步续跑，最多约 **20,160 次策略回合请求**，再加至多 96 个 proposer 请求及有限重试。续跑通常短于 15 步，但不得据此承诺工期。

若输出 cap 为 512，策略回合输出 token 上限约 1,032 万；累计 prompt token 另计，可能更高。该上限是预算估算，不是实际成本。

必须实现 `--max-api-calls`、`--max-generated-tokens`、`--max-walltime-hours`。为防超预算，发请求前预留其 max_tokens；并发初版=1，后续做 semaphore 预留。不以超预算后的裁剪样本冒充预定无偏评估。

P1 后按实际 episodes/sec、turns/episode、p50/p95 latency 和 token 测量估算 P2 工期；旧 HotpotQA 约 13–23 秒/题不能直接外推 MuSiQue 长上下文。

### 待实现 CLI 契约（当前不可直接运行）

```text
python scripts/audit_runtime_efficiency.py --inputs <paths...> --output-dir <run>/reports
python scripts/runtime_efficiency_pilot.py prepare --tasks-dir <train> --corpus-dir <train_corpus> --selection-seed 20260929 --output-dir <run>
python scripts/runtime_efficiency_pilot.py collect --run-dir <run> --stage smoke --model-url <url> --model-name <id> --max-api-calls <cap> --max-generated-tokens <cap> --max-walltime-hours <cap> --resume
python scripts/runtime_efficiency_pilot.py prepare-branches --run-dir <run> --stage smoke
python scripts/runtime_efficiency_pilot.py continue --run-dir <run> --stage smoke --model-url <url> --model-name <id> --max-api-calls <cap> --max-generated-tokens <cap> --max-walltime-hours <cap> --resume
python scripts/runtime_efficiency_pilot.py analyze --run-dir <run>
```

Luna 实现后必须用实际 `--help` 和 CPU smoke 验证 CLI，再给用户可复制的服务器命令。不同阶段预算、重复数写入 protocol；resume 不允许隐式从 smoke 扩成完整实验。

## 11. 后续训练研究：仅在诊断通过后另立计划

当前不选定算法名称，也不承诺 A 会录用。

若确认存在可学习的便宜路径，先比较简单基线：

1. 固定成本权重的 cost-aware GRPO（小范围权重 sweep，统一预算）。
2. 约束式/自适应乘子方法（参考 LACONIC/CCPO 的思想与适用条件）。
3. 经独立验证的短成功轨迹 SFT 或效率偏好学习（参考 DEPO）。
4. 若新贡献涉及训练分支选择，再加入均匀分支、BPO 风格熵分支、EPIG 风格梯度信息分配；否则不必把所有树方法堆进首轮诊断。

必须区分训练总开销和部署开销，公平核算额外数据生成。后续训练至少 3 个训练随机种子；多个推理 seed 不能替代多训练 seed。先在两个设置稳定基线，再对多个独立设置验证拟议贡献及消融，不在首轮 pilot 承诺全面 benchmark。

有意义的新颖性可能来自“识别了既有方法失败的具体机制并解决”，不是把 CMDP、DPO、分支采样换名字拼起来。若证据仅支持已知方法有效，诚实记录为工程/复现结果。

## 12. 近邻阅读与结论边界

- [VinePPO](https://arxiv.org/abs/2410.01679)：中间状态 MC value estimation。
- [BPO](https://arxiv.org/abs/2607.14171)：Agent 中间快照、选择分支、兄弟回报基准。
- [InfoTree](https://arxiv.org/abs/2605.05262)：固定预算下中间状态选择与预算分配。
- [EPIG-Tree](https://arxiv.org/abs/2609.20004)：梯度相关信息与计算分配；局部估计质量有适用边界。
- [LACONIC](https://arxiv.org/abs/2602.14468)：约束式长度控制、自适应成本系数。
- [CCPO，AAAI 正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/39739)：可靠性约束下 Agent 成本优化。
- [DEPO](https://arxiv.org/abs/2511.15392)：每步 token 与轨迹步数的效率偏好优化。

这些链接用于确认重合问题和实验基线，不表示其全部理论/实验已经独立复现。不要把优势方差、梯度向量方差和梯度范数方差混为一谈；也不要把统计相关性改写成“因果冗余”。

## 13. 可直接发给 Luna 的交接提示

> 请按 `docs/runtime_efficiency_experiment_plan_20260929.md` 实现冻结策略的运行时效率诊断。先读工作区与现有 runner/state/collector/client，保留未提交改动。先做 P0 和 CPU 单元测试，再提供/执行 P1 smoke；P1 验收通过后运行受预算限制的 P2。核心是完整前缀恢复、KEEP_A/REPLAN/ALT_PREFIX_ONLY 的正确语义、无 gold/未来信息泄漏、独立验证、双成本账本和断点续跑。不要直接训练，不要默认占用 GPU，不要停别人的服务，不要擅自 push。若没有服务器访问，交付完整运行命令并明确尚未运行。最终给出测试证据、真实运行清单、统计区间、失败案例和 go/no-go 结论；数据不支持假设时也必须如实报告。
