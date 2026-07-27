# ResearchAgent-RL 实验记录

最后更新：2026-07-22。此文档是项目的离线实验台账；原始大文件继续保存在服务器 `/data/zhangyonglin/research-agent-rl-data`，仓库只保存配置、指标、日志位置和结论。

## 记录规范

每个可比较实验必须固定并记录：代码提交、模型/检查点、数据集与哈希、随机种子、GPU、训练/服务参数、输出目录，以及汇总指标。评测对比优先使用同一批任务和同一个 `selection_seed`。

HotpotQA 主指标排序如下：

1. `task_success`
2. `answer_quality`、`citation_f1`
3. `action_parse_success_rate`（越高越好）和 `invalid_action_rate`（越低越好）
4. `average_steps`、延迟和 token 数（成本诊断，不作为主效果指标）

## 数据集

| 项目 | 值 |
| --- | --- |
| 来源 | `hotpot_qa` / `distractor` |
| 切分 | 7,000 train / 3,000 eval，70% / 30% |
| train / eval 语料块 | 69,718 / 29,838 |
| 隔离性 | task ID 无重叠；语料按 split 隔离；检索按任务参考文档限定；全部标注引用存在 |
| 服务器根目录 | `/data/zhangyonglin/research-agent-rl-data/hotpotqa_7k3k` |

严格动作 SFT 数据：

| 项目 | 值 |
| --- | --- |
| 文件 | `/data/zhangyonglin/research-agent-rl-data/sft_data/hotpotqa_train_7k_strict_actions.jsonl` |
| 记录数 | 7,000 |
| SHA-256 | `90090894129eed1ade01cbcdbdb73f5f98044f03e6fe277a720035e32ada4249` |
| 格式 | JSONL；标准 `system` / `user` / `assistant` chat messages |
| 监督轨迹 | `SEARCH → READ → CITE → ANSWER`，每题 4 个 assistant action |
| 检索约束 | `SEARCH` 使用问题本身作为 query，并返回覆盖支持证据的任务内候选集合；`READ` 只读取其中支持块 |

该 manifest 已于 2026-07-22 从服务器导入本机项目 `artifacts/sft7k/` 并核对一致（该本机 artifact 不纳入 Git）。

## 已完成实验

### E01：基座模型完整评测

| 字段 | 值 |
| --- | --- |
| 模型 | `Qwen3.5-9B` |
| 评测集 | HotpotQA eval 全量 3,000 |
| 输出 | `~/ResearchAgent-RL/results/hotpotqa_eval_base_full_seed20260713` |
| 运行状态 | 已完成 |

| 指标 | 值 |
| --- | ---: |
| task success | 0.3207 |
| answer quality | 0.3404 |
| answer contains | 0.3260 |
| answer token F1 | 0.0854 |
| citation precision / recall / F1 | 0.4397 / 0.3050 / 0.3493 |
| action parse success | 0.8809 |
| invalid action rate | 0.1358 |
| average steps | 5.269 |
| average latency / episode | 23.02 s |

这是后续模型的全量基线。

### E02：基座模型固定 100 题 smoke

| 字段 | 值 |
| --- | --- |
| 模型 | `Qwen3.5-9B` |
| 题目 | eval 中 100 题，`selection_seed=20260713` |
| 输出 | `/data/zhangyonglin/research-agent-rl-data/outputs/hotpotqa_eval_base_seed20260713_n100_rerun2` |
| 运行状态 | 已完成 |

| 指标 | 值 |
| --- | ---: |
| task success | 0.3000 |
| answer quality | 0.3134 |
| citation F1 | 0.3070 |
| action parse success | 0.9130 |
| invalid action rate | 0.0970 |
| average steps | 5.35 |
| average latency / episode | 23.45 s |

此固定 100 题集合是检查点筛选的低成本标准对照。

### E03：Vime 磁盘权重同步稳定性

| 字段 | 值 |
| --- | --- |
| 输出 | `/data/zhangyonglin/research-agent-rl-data/outputs/vime_disk_sync_stability5_keep_20260720_141953` |
| 运行状态 | 已完成，Ray job succeeded |
| 证据 | 6 次 `weight_v000001`–`weight_v000006` 均被 vLLM 重载 |
| 单次 vLLM 重载时间 | 2.83–3.00 s |
| 备注 | 保留同步文件的调试运行占用约 101 GB；正式运行应删除或不保留历史同步权重。 |

结论：当前 vLLM 0.23 兼容模式的磁盘同步链路在独立稳定性实验中已验证，不应再使用此前失败的内存权重同步路径。注意：GRPO100 的 `driver.log` 同时出现大量 `Failed to load weights` 警告（主要来自 `Worker_TP1`）；尽管随后有成功的 checkpoint 保存与重载记录，仍应把它视为该历史运行的基础设施风险，不能据此单独断言同步实现完全无误。

### E04：GRPO 100 step（失败/回归对照）

| 字段 | 值 |
| --- | --- |
| 训练输出 | `/data/zhangyonglin/research-agent-rl-data/outputs/vime_hotpotqa_grpo100_20260720_151526` |
| 最终权重 | `weight_v000101`（同时保留 `weight_v000100`） |
| 后评测 | `~/ResearchAgent-RL/results/hotpotqa_eval_rl100_seed20260713_n100` |
| 题目 | 固定 100 题，`selection_seed=20260713` |
| 运行状态 | 已完成，但明显回归 |

| 指标 | 基座 n=100 | GRPO100 n=100 |
| --- | ---: | ---: |
| task success | 0.3000 | **0.1100** |
| answer quality | 0.3134 | **0.0986** |
| citation F1 | 0.3070 | **0.1533** |
| action parse success | 0.9130 | **0.3987** |
| invalid action rate | 0.0970 | **0.6013** |
| average steps | 5.35 | 4.42 |

诊断结论：训练后出现动作格式坍缩（解释性文字、无 action 或多个 action），而不是基准评测脚本失效。训练日志中的 reward 约 1.4–1.96 是组内中心化前的原始奖励；GRPO advantage 接近 0 是 group-centering 的正常现象。该运行的 `--kl-loss-coef 0.00` 没有提供 KL 约束。且该历史日志存在前述 TP1 权重加载告警，所以它应被视为“策略/格式约束不足与可能的权重加载风险共同存在”的负对照，而非单因果实验。**禁止把该 GRPO 权重作为候选模型继续训练。**

### E05：SFT smoke（32 题）

| 字段 | 值 |
| --- | --- |
| 有效运行 | `/data/zhangyonglin/research-agent-rl-data/outputs/vime_hotpotqa_sft_smoke32_fixed_20260721_155855` |
| 数据 | 上述 SFT 数据的前 32 条 |
| 结果 | 成功，保存至 `iter_0000003` |
| 训练 loss | 约 0.439 → 0.088 |

较早的 SFT smoke 因数据包含非标准 `observation` role 而失败；该失败运行不用于比较。

### E06：全量严格动作 SFT（7k）

| 字段 | 值 |
| --- | --- |
| 输出 | `/data/zhangyonglin/research-agent-rl-data/outputs/vime_hotpotqa_sft7k_20260721_162728` |
| 数据 | 7,000 条严格动作 SFT 数据 |
| 更新数 | 875（step 0–874） |
| 运行状态 | 已完成，Ray job succeeded |
| 最终 checkpoint | `iter_0000874` |
| 保留 checkpoints | `iter_0000249`、`iter_0000499`、`iter_0000749`、`iter_0000874`、`latest_checkpointed_iteration.txt` |
| 目录大小 | 约 540 GB（检查点包含优化器状态） |

最终训练端指标：

| 指标 | 值 |
| --- | ---: |
| `train/loss`（step 874） | 0.0067243 |
| grad norm（step 874） | 0.3369 |
| learning rate（step 874） | 1e-6 |
| step time（step 874） | 21.35 s |

训练曲线抽样（同一 `driver.log`）：

| step | train loss | grad norm | learning rate |
| ---: | ---: | ---: | ---: |
| 0 | 0.44937 | 39.983 | 3.81e-7 |
| 249 | 0.00629 | 0.439 | 8.54e-6 |
| 499 | 0.00523 | 0.368 | 4.68e-6 |
| 749 | 0.00688 | 0.373 | 1.47e-6 |
| 874 | 0.00672 | 0.337 | 1.00e-6 |

说明：此 SFT 启动配置为 `--debug-train-only`，因此日志中的 `rollout/rewards=0` 是预期行为，不能用作任务质量指标；step 0 与 step 874 的 `rollout/truncated` 都是 0。任务质量仍必须经独立 HotpotQA rollout 评测得出。

2026-07-22 核对的 checkpoint 清单确认 `latest_checkpointed_iteration.txt=874`，并且四个待评目录均存在；`rollout/` 目录不是模型 checkpoint。

## SFT checkpoint 筛选（固定 n=100，已完成）

不能仅凭最低 SFT loss 选择模型。四个 checkpoint 已使用同一固定 100 题和 `selection_seed=20260713` 完成独立环境评测：

| checkpoint | task success | answer quality | citation F1 | parse success | invalid rate | 平均步骤 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iter_0000249` | 0.760 | 0.746 | 0.860 | 1.000 | 0.003 | 4.02 | 12.87 s |
| `iter_0000499` | 0.770 | 0.764 | 0.900 | 1.000 | 0.000 | 4.00 | 12.80 s |
| `iter_0000749` | 0.770 | 0.743 | 0.905 | 1.000 | 0.000 | 4.00 | 12.80 s |
| `iter_0000874` | **0.800** | **0.781** | **0.925** | 1.000 | 0.000 | 4.00 | 12.78 s |

| 项目 | 值 |
| --- | --- |
| 评测输出 | `~/ResearchAgent-RL/results/hotpotqa_eval_sft_ckpts_n100_20260722_135303` |
| 运行状态 | 四个 checkpoint 均完成 |
| 选择结果 | **`iter_0000874`** |

结论：所有 SFT checkpoint 都消除了基座/GRPO100 中的主要格式问题，未观察到后期 checkpoint 坍缩。`iter_0000874` 同时取得最高 task success、answer quality 和 citation F1，故作为唯一候选进入完整 3,000 条评测。相对于固定 n=100 基座，`iter_0000874` 的 task success 从 0.300 提升至 0.800，parse success 从 0.913 提升至 1.000，invalid rate 从 0.097 降至 0。

## SFT 最优 checkpoint 的完整评测（已完成）

`iter_0000874` 已完成全量 3,000 题评测，并已与 E01 基座结果比较。它保持了 n=100 筛选中的领先表现，因而满足进入下一阶段、带 KL 约束和格式监控的 GRPO 的前置条件。

### E07：SFT-874 完整评测（3,000 条）

| 字段 | 值 |
| --- | --- |
| 模型 | SFT `iter_0000874` |
| 评测集 | HotpotQA eval 全量 3,000，`selection_seed=20260713` |
| 输出 | `~/ResearchAgent-RL/results/hotpotqa_eval_sft874_full_seed20260713_20260722_154042/iter_0000874/eval` |
| 运行状态 | 已完成；2026-07-22 15:40 至 2026-07-23 02:28 |

| 指标 | 基座全量（E01） | SFT-874 全量（E07） | 变化 |
| --- | ---: | ---: | ---: |
| task success | 0.3207 | **0.7913** | +0.4707 |
| answer quality | 0.3404 | **0.7728** | +0.4324 |
| citation F1 | 0.3493 | **0.9353** | +0.5860 |
| action parse success | 0.8809 | **0.9996** | +0.1186 |
| invalid action rate | 0.1358 | **0.0020** | −0.1338 |
| average steps | 5.269 | **4.010** | −1.259 |
| average latency / episode | 23.02 s | **12.87 s** | −10.15 s |

结论：SFT-874 的全量结果确认固定 n=100 筛选并非偶然；它在正确性、引用质量、格式可靠性和延迟上均显著优于基座。该 checkpoint 通过进入下一阶段 GRPO 的效果门槛，但新的 GRPO 实验必须保留非零 KL 约束、逐轮格式监控和固定 n=100 checkpoint 评测，避免重现 E04 的退化。

仓库脚本 `scripts/evaluate_sft_checkpoints.sh` 固化了上述四 checkpoint 的顺序导出、健康检查、单 GPU vLLM 服务和固定 n=100 评测流程；默认输出在 `~/ResearchAgent-RL/results/`，并允许通过环境变量覆盖路径、端口和 GPU。

## 训练日志中应看的内容

### SFT

从 `driver.log` 解析并记录：`train/loss` 曲线、grad norm、学习率、吞吐/step time、保存 checkpoint 行、最终 Ray job 状态。loss 只反映模仿监督动作的拟合程度；选择模型必须结合 rollout 指标。

```bash
SFT_RUN=/data/$USER/research-agent-rl-data/outputs/vime_hotpotqa_sft7k_20260721_162728
grep 'model.py:909 - step ' "$SFT_RUN/driver.log" > "$SFT_RUN/sft_train_metrics.log"
tail -n 120 "$SFT_RUN/driver.log"
```

### GRPO（只在 SFT 筛选通过后）

额外记录：每组原始 reward 与方差、KL、entropy、clip fraction、policy/critic loss、格式成功率、每轮权重同步是否成功、训练中和训练后的固定评测。任何 parse success 明显下降或 invalid rate 上升的 checkpoint 都应停止扩展训练。

## 如何把服务器结果交给我更新此文档

优先顺序如下，不需要复制大模型或整套 checkpoint：

1. 直接把终端输出、`metrics_summary.json` 内容和 `driver.log` 的关键片段粘贴到此对话。
2. 将小型产物下载到本机后作为附件发给我。macOS 示例：

```bash
mkdir -p ~/Downloads/research-agent-rl-artifacts
scp zhangyonglin@172.22.0.45:~/ResearchAgent-RL/results/<run>/metrics_summary.json \
  ~/Downloads/research-agent-rl-artifacts/
scp zhangyonglin@172.22.0.45:/data/zhangyonglin/research-agent-rl-data/outputs/<run>/driver.log \
  ~/Downloads/research-agent-rl-artifacts/
```

3. 若日志较大，在服务器上先压缩所需文本再下载：

```bash
ssh zhangyonglin@172.22.0.45 \
  'tar -C /data/zhangyonglin/research-agent-rl-data/outputs/<run> \
  -czf /tmp/<run>-logs.tar.gz driver.log sft_train_metrics.log'
scp zhangyonglin@172.22.0.45:/tmp/<run>-logs.tar.gz \
  ~/Downloads/research-agent-rl-artifacts/
```

每次提交给我时，请同时给出：运行目录、代码 commit、命令/关键超参数、`metrics_summary.json`、日志末尾 100 行，以及要比较的基线名称。我会更新本台账并推送到 GitHub。不要传输 100GB 级权重或含凭证的配置文件。
