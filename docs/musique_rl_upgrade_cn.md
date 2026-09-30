# MuSiQue 多跳检索与 Agent RL 数据协议

## 为什么选 MuSiQue

原先的 HotpotQA distractor 训练集把每题约 10 个候选段落放在一个 task document 中。模型可以一次 `SEARCH(topk=10)` 看见整个候选池，检索策略很难成为 RL 的主要学习对象。HotpotQA 官方没有名为“HotpotQA+”的独立版本；官方设置是 distractor 和 fullwiki。新版选择 [MuSiQue-Answerable v1.0](https://github.com/stonybrooknlp/musique)：它提供 2–4 跳问题、逐跳证据段落和官方 train/dev 划分，并专门降低单跳捷径的可解性。另一个可选更难集合 [MoreHopQA](https://github.com/Alab-NII/morehopqa) 只有 1,118 个经人工核验的样本，更适合作外部测试，不适合担任这里的主要 RL 训练集。

`eval` 对应 MuSiQue 官方 **dev**，有答案，可以本地评分；官方 test 隐藏标签，不能称本地 `eval` 为官方 test 结果。原 HotpotQA 的分数与新版不能直接比较。

## 数据与索引

`scripts/prepare_musique.py` 将官方 JSONL 转成：

```text
<output_dir>/
  tasks/train/*.json
  tasks/eval/*.json
  corpus/train/corpus.sqlite
  corpus/eval/corpus.sqlite
  stats.json
```

每题的 `ground_truth_citations` 是支持段落的 chunk ID；`analysis.hops` 保存逐跳问题和支持段落 ID，只用于离线诊断，不进入 Agent 提示词。`retrieval_scope=split_corpus` 告诉运行环境在整个 split 的池化语料中检索，`reference_docs=[]` 不再把搜索缩回单题候选池。train/eval 分别有自己的 SQLite FTS5 索引，按标题和段落正文做 BM25 检索，标题权重为 5。模型可以从 READ 到的实体构造下一次 SEARCH query。

为减少严格评测中的直接证据泄漏，构建 train 索引时会删除与 eval 支持段落完全相同的 train distractor；如果 train 的 gold 支持段落与 eval gold 完全相同，则构建直接报错。这里的“相同”按标题和段落正文的精确哈希判断，不能排除改写后的语义重叠或基座预训练知识。

当前本机生成结果：19,938 条 train、2,417 条 eval；train 84k 左右、eval 21,100 个去重段落；693 个与 eval gold 相同的 train distractor 被排除。train/eval gold 支持段落交集为 0。源文件约 270 MB，生成数据加索引约 203 MB。生成文件在项目的 `data/musique_rl_v2/`，由 `.gitignore` 排除，不会意外推送数据集。

## 在服务器 `/data` 生成

先更新包含这些脚本的项目代码，再运行。以下镜像为固定 revision；原始发布页是 [MuSiQue 官方仓库](https://github.com/stonybrooknlp/musique)，许可证为 CC BY 4.0。

```bash
cd ~/ResearchAgent-RL
conda activate /data/$USER/conda_envs/vime-train-cu129

BASE="/data/$USER/research-agent-rl-data"
SOURCE="$BASE/musique_source"
BENCH="$BASE/musique_rl_v2"
mkdir -p "$SOURCE"

curl -fL --retry 5 --retry-all-errors -o "$SOURCE/musique_ans_v1.0_train.jsonl" \
  'https://huggingface.co/datasets/voidful/MuSiQue/resolve/a7d9f9adf6191604fc67cde318ee1a86fcf7babc/musique_ans_v1.0_train.jsonl'
curl -fL --retry 5 --retry-all-errors -o "$SOURCE/musique_ans_v1.0_dev.jsonl" \
  'https://huggingface.co/datasets/voidful/MuSiQue/resolve/a7d9f9adf6191604fc67cde318ee1a86fcf7babc/musique_ans_v1.0_dev.jsonl'

sha256sum "$SOURCE"/*.jsonl
python scripts/prepare_musique.py --source_dir "$SOURCE" --output_dir "$BENCH"
python scripts/prepare_slime_dataset.py \
  --tasks_dir "$BENCH/tasks/train" --output_file "$BENCH/prompts/train.jsonl"
python scripts/prepare_slime_dataset.py \
  --tasks_dir "$BENCH/tasks/eval" --output_file "$BENCH/prompts/eval.jsonl"
python scripts/evaluate_retrieval.py \
  --tasks_dir "$BENCH/tasks/eval" --corpus_dir "$BENCH/corpus/eval" \
  --topk 5 10 20 --output_file "$BENCH/retrieval_eval.json"
```

预期 SHA-256：train `83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a`；dev `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`。构建器拒绝覆盖已有 output directory；重跑请指定新的目录名。

## 检索诊断和后续训练

本机完整 2,417 条 eval 上的初步检索诊断：

| topk | 原问题第一跳召回 | 原问题全部支持段落召回 | 用 gold 中间答案改写的逐跳召回 | oracle 完整链召回 |
|---:|---:|---:|---:|---:|
| 5 | 0.7849 | 0.1175 | 0.8448 | 0.6798 |
| 10 | 0.8502 | 0.1692 | 0.8963 | 0.7741 |
| 20 | 0.8941 | 0.2313 | 0.9319 | 0.8382 |

“原问题全部支持段落召回”是一次 SEARCH 同时找到所有跳；它偏低正是 Agent 需要逐跳搜索的原因。`oracle` 诊断使用标签中的中间答案，只衡量检索器上限，不代表模型推理能力，也不进入训练或评测提示词。

在固定种子抽取的 1,000 条 train 上，第一跳 Recall@10 为 0.829，oracle 完整链 Recall@10 为 0.651。训练池更大，逐跳检索本身仍会失败；RL 最终正确率不可能仅靠优化决策策略达到 oracle 上限以上。因此正式训练前应先看模型生成的第二跳 query 和检索命中率，必要时进一步升级混合召回。

开始 GPU 实验前，用新基座/SFT 初始化模型在固定 100 条 eval 上跑 `--max_steps 12`，报告 task success、citation recall、parse success 和每跳召回。MuSiQue 的任务成功要求答案质量过线且**所有 gold 支持段落都被引用**。RL reward 也按 citation recall 缩放答案奖励，减少“猜中答案但只引用一跳”的捷径。

现有 Vime 启动器支持通过 `PROMPT_DATA`、`CORPUS_DIR` 和 `RESEARCH_AGENT_MAX_STEPS` 接入新数据；4 跳任务的理想路径约需四次 SEARCH、四次 READ、CITE、ANSWER，共 10 步。设为 12 步可容纳少量查询修正。先做一个 rollout smoke，检查样本长度、显存、权重同步和有效动作率，再定正式训练步数：

```bash
RUN_DIR="$BASE/outputs/vime_musique_grpo_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
GPU_IDS=2,3,4,5,6,7 \
PROMPT_DATA="$BENCH/prompts/train.jsonl" \
CORPUS_DIR="$BENCH/corpus/train" \
RESEARCH_AGENT_MAX_STEPS=12 \
NUM_ROLLOUT=1 \
RUN_DIR="$RUN_DIR" \
nohup bash scripts/run_vime_hotpotqa_sft_grpo_v2.sh \
  >"$RUN_DIR/driver.log" 2>&1 < /dev/null &
echo $! >"$RUN_DIR/pid.txt"
```

现有 SFT-874 学的是 HotpotQA 四步轨迹，它对反复搜索的迁移效果未知，必须先测；若有效动作或多跳完成率低，再做 MuSiQue 多跳轨迹的冷启动 SFT。`NUM_ROLLOUT=1` 只做协议烟测，不提供训练效果结论。完整训练和基座评测都需要服务器 GPU；本机只完成了数据构建、检索诊断和代码单测。
