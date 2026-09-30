# Effect-space 判别实验：服务器运行手册

状态：代码和本地单元测试已准备；**还没有在服务器模型上运行，也没有任何效果结果。** 先跑 MuSiQue train 的低成本诊断，不启动 Vime/GRPO。不要使用官方 eval/dev 挑选效果类或调参。

## 0. 从 GitHub 获取代码，单独同步数据

2026-09-30 用户选择通过提交/推送和服务器拉取同步代码。35的推理环境已迁移并通过GPU kernel验证，项目目标路径为`/local_data/zhangyonglin/ResearchAgent-RL`。以下在**35**执行；已有Git工作区须位于`slime-rewrite`且没有跟踪文件改动，非Git非空目录不会被覆盖。

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
if [ ! -d "$PROJECT/.git" ]; then
  git clone --branch slime-rewrite --single-branch \
    https://github.com/wishfine/ResearchAgent-RL.git "$PROJECT"
else
  test "$(git -C "$PROJECT" branch --show-current)" = slime-rewrite
  git -C "$PROJECT" diff --quiet
  git -C "$PROJECT" diff --cached --quiet
  git -C "$PROJECT" pull --ff-only origin slime-rewrite
fi
git -C "$PROJECT" log -1 --oneline
)
```

MuSiQue数据不入Git；以下在**本机项目目录**执行，只同步数据。模型继续从45传到35，环境与模型文件均不进仓库。

```bash
cd /Users/wishfine/Documents/RL/RL_project/ResearchAgent-RL
ssh xdf-35 'mkdir -p /local_data/zhangyonglin/ResearchAgent-RL/data/musique_rl_v2'
rsync -avh --partial --progress data/musique_rl_v2/ \
  xdf-35:/local_data/zhangyonglin/ResearchAgent-RL/data/musique_rl_v2/
```

## 1. 环境检查

以下均在**35**执行。使用独立迁移的`research-agent-runtime`环境。根据实际服务修改地址与模型名。必须记录服务背后的不可变checkpoint/权重路径；`/v1/models`只能核对served name，脚本不能自动证明实际权重就是该路径。

```bash
PROJECT="/local_data/$USER/ResearchAgent-RL"
cd "$PROJECT"
conda activate "/local_data/$USER/conda_envs/research-agent-runtime"
BASE="/local_data/$USER/research-agent-rl-data"
BENCH="$PROJECT/data/musique_rl_v2"
MODEL_URL="http://127.0.0.1:8001/v1"
MODEL_NAME="Qwen3.5-9B-SFT874"
MODEL_REVISION="/local_data/$USER/models/ResearchAgent-Qwen3.5-9B-SFT874"
test -d "$BENCH/tasks/train" && test -f "$BENCH/corpus/train/corpus.sqlite"
curl -fsS "$MODEL_URL/models" | python -m json.tool
python -m pytest -q tests/test_effect_space.py tests/test_effect_space_continuation_pilot.py
```

`MODEL_REVISION`必须指向实际服务权重。迁移环境若缺少测试依赖，先记录缺项；不要升级torch/vLLM来解决CPU测试依赖。运行此手册需先启动对应模型服务，本手册不会自行启动服务。

## 2. 首轮 SEARCH 采样（先小 smoke，再独立完整 pilot）

小 smoke 检验 API、解析器、全库检索及落盘。20 题可能没有足够碰撞，不据此否定方向；完整 pilot 使用新目录。运行期间日志会逐题更新。假设模型服务已启动，脚本不会自行启动 vLLM。

```bash
SMOKE="$BASE/outputs/effect_space_first_smoke20_n8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE"
python -u scripts/effect_space_pilot.py \
  --tasks_dir "$BENCH/tasks/train" \
  --corpus_dir "$BENCH/corpus/train" \
  --model_url "$MODEL_URL" --model_name "$MODEL_NAME" \
  --model_revision "$MODEL_REVISION" \
  --max_tasks 20 --samples_per_task 8 --selection_seed 42 \
  --temperature 1.0 --top_p 0.95 --max_tokens 256 \
  --output_dir "$SMOKE" >"$SMOKE/driver.log" 2>&1
cat "$SMOKE/summary.json"
```

```bash
FIRST="$BASE/outputs/effect_space_first_train200_n8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FIRST"
nohup python -u scripts/effect_space_pilot.py \
  --tasks_dir "$BENCH/tasks/train" \
  --corpus_dir "$BENCH/corpus/train" \
  --model_url "$MODEL_URL" --model_name "$MODEL_NAME" \
  --model_revision "$MODEL_REVISION" \
  --max_tasks 200 --samples_per_task 8 --selection_seed 42 \
  --temperature 1.0 --top_p 0.95 --max_tokens 256 \
  --output_dir "$FIRST" >"$FIRST/driver.log" 2>&1 < /dev/null &
echo $! >"$FIRST/pid.txt"
echo "FIRST=$FIRST"
```

跨 shell 要用 `FIRST` 的**实际绝对路径重新赋值**，否则空变量会指向 `/driver.log`。`tail -f "$FIRST/driver.log"` 看进度；`summary.json` 出现并能解析才算采样+召回诊断完成。中断后在**原目录、原参数**加 `--resume` 续跑，不要混用其他模型服务或 checkpoint。若第一步的模型版本变化，必须新开 run。

## 3. 配对 continuation：先 prepare 后 run

只在 `FIRST/summary.json` 生成后继续。`prepare` 仅用任务文本、样本动作和检索结果选对，不读 gold answer/citations；输出配对清单和输入哈希。先看 `selected_tasks`、`eligible_tasks`，若没有同效果的不同 query，对精确效果空间路线先停机，不运行下游生成。

```bash
PAIR="$BASE/outputs/effect_space_continuation_train10_r6_$(date +%Y%m%d_%H%M%S)"
python scripts/effect_space_continuation_pilot.py prepare \
  --tasks_dir "$BENCH/tasks/train" \
  --corpus_dir "$BENCH/corpus/train" \
  --samples_file "$FIRST/samples.jsonl" \
  --output_dir "$PAIR" \
  --max_tasks 10 --max_pairs_per_task 1 --selection_seed 42
python -m json.tool "$PAIR/manifest.json" | head -n 45
```

`PAIR` 的正式 run 采用每个 query 6 次 continuation、两种历史模式。这仍只是约 10 题的链路 pilot，不能当论文显著性结果。`--max_steps 12` 包含已经强制执行的第一步 SEARCH；该步不计新的模型生成 token，但总步数和奖励照实计算。

```bash
mkdir -p "$PAIR"
nohup python -u scripts/effect_space_continuation_pilot.py run \
  --output_dir "$PAIR" \
  --model_url "$MODEL_URL" --model_name "$MODEL_NAME" \
  --model_revision "$MODEL_REVISION" \
  --repeats 6 --max_steps 12 --max_tokens 256 \
  --temperature 0.7 --top_p 0.95 \
  >"$PAIR/driver.log" 2>&1 < /dev/null &
echo $! >"$PAIR/pid.txt"
echo "PAIR=$PAIR"
```

每题原子落盘 `task_runs/<task_id>.json`；中断后沿用**完全相同参数**并加 `--resume`，脚本会拒绝混合参数。若首轮采样与 continuation 不是同一模型版本、语料库或任务文件变了，也不要复用该 manifest。脚本校验语料索引、所选 task、样本的 SHA-256，以及服务模型名；`model_revision` 仍需由操作者如实填入。

## 4. 看结果、决定是否继续

```bash
tail -n 50 "$PAIR/driver.log"
python scripts/effect_space_continuation_pilot.py analyze --output_dir "$PAIR"
python -m json.tool "$PAIR/summary.json"
```

重点同时看 `mean_within_original_reward_gap`、`mean_between_original_reward_gap`、`mean_within_canonical_reward_gap` 与 `mean_original_split_half_reward_noise_floor`，以及逐对的 `max_retrieval_score_shift`。差值只是有限次生成的**描述性绝对均值差**，有上偏；bootstrap 区间以题目为单位，少于 5 题时不报告。`canonical` 只改模型所见的首轮 assistant action 文本，环境仍用原 query；它不能被解释为当前部署策略的收益。若同类差没有小于异类差、与噪声尺度无法区分，或者碰撞率低，先不实现权重复用或 RL 更新。

需要带回本机复核的最小文件：`FIRST/sampling_config.json`、`FIRST/samples.jsonl`、`FIRST/summary.json`、`FIRST/effect_records.jsonl`、`PAIR/manifest.json`、`PAIR/run_config.json`、`PAIR/summary.json`、`PAIR/driver.log`，再抽取少量 `PAIR/task_runs/*.json` 供失败案例审阅。原始采样可能含任务和模型输出，不要未经检查就公开推送。
