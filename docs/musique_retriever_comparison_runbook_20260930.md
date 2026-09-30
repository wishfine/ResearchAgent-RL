# 冻结SFT874：BM25 vs Qwen hybrid Agent对照

## 要验证什么

离线1000题原问题Top-10全证据召回由BM25的20.5%提高到Qwen hybrid的29.1%。本轮验证这个环境升级是否提高**实际多轮Agent**的答案/引用质量，以及增加多少检索、阅读、输入输出token与端到端延迟。不训练，不宣称RL或新算法收益，不以Oracle查询喂策略。

使用35已有的冻结SFT874服务（GPU0、8105、32K）及Qwen embedding（8107，用户要求与BGE共GPU1）。入口不启动/关闭服务，不跑Ray，不需要闲置的训练卡。BGE不参与本轮。

## 固定实验矩阵

24道train题（2/3/4-hop各8道），选样seed20260929；每题2次采样重复。每个检索器运行以下4组，总计192 episode，两检索器384 episode：

| 组 | prompt | READ |
|---|---|---|
| cite_first_head300 | cite-first协议 | 正文头300字符 |
| cite_first_full | 同上 | 全文 |
| adaptive_head300 | 自适应多跳协议 | 正文头300字符 |
| adaptive_full | 同上 | 全文 |

两检索器使用完全相同的task文件SHA、job顺序、每job/每turn种子、模型文件SHA、代码、语料、prompt/READ条件与采样参数：max_steps15、max_tokens512/动作、temperature0.7、top_p0.95。只改变检索配置。prepare阶段比较全部manifest字段（retrieval除外），包括显式检查Qwen索引和语料SHA一致；不匹配则在模型调用前停止。随后在任何生成调用前，检查策略/embedding的实际API身份、模型及语料文件SHA、索引完整文件checksum/shape；证据保存在`preflight.json`。坏索引或未就绪的Qwen服务不会等到BM25完成才报错。检查不发generation/embedding POST，不修改核心protocol客户端。

模型服务必须冻结、不外部reload；相同采样seed不保证GPU推理逐比特可复现。两检索器顺序执行，避免同一生成服务竞争；先BM25后Qwen，墙钟/系统负载比较仍可能存在顺序效应。探索阶段发现明显效果后，可再做交换顺序重复。

每检索器 policy上限2880次API请求、1,474,560生成token、每次进程调用4小时；合计最多5760次/2,949,120生成token。embedding另记账，不包含在policy token上限内；没有专门的embedding token上限。wall限制是单次调用预算，不是跨resume累计硬期限，检查在请求前执行，在途请求可能超过期限。不宣称8小时为完整硬总预算或预计时长。

## 35运行命令

先拉取代码并确认两个已有接口，不重启服务：

```bash
(
set -euo pipefail
test "$(hostname)" = "m7-2-5-a1-7-29U-AI"
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
cd "$PROJECT"
git pull --ff-only origin slime-rewrite
curl -fsS --max-time 10 http://127.0.0.1:8105/v1/models
curl -fsS --max-time 10 http://127.0.0.1:8107/v1/models
COMPARE_ROOT="$BASE/runtime_efficiency/musique_retriever_compare24_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMPARE_ROOT"
CUDA_VISIBLE_DEVICES="" COMPARE_ROOT="$COMPARE_ROOT" \
  nohup bash scripts/run_musique_retriever_compare.sh \
  > "$COMPARE_ROOT/driver.log" 2>&1 < /dev/null &
echo $! > "$COMPARE_ROOT/pid.txt"
printf '%s\n' "$COMPARE_ROOT" > "$BASE/runtime_efficiency/latest_retriever_compare.txt"
echo "COMPARE_ROOT=$COMPARE_ROOT"
)
```

入口生成项目`artifacts/runtime_efficiency/<run>`软链和两个子目录`bm25/`、`qwen_hybrid/`，直接调用现有Python客户端，不生成全局`artifacts/runtime_efficiency/bm25`等冲突软链。并发启动同一个目录时flock拒绝第二个driver。相同run恢复会检查原manifest、跳过完成job、保存失败旧attempt，累计policy/embedding成本，并追加日志。若模型/代码/配置/数据已变化，必须新建run，不能拼接旧轨迹。

可在前台设置`PREPARE_ONLY=1`、`COMPARE_ROOT=<新目录>`运行同一入口，仅写清单和核对manifest、不联系API、不执行完整索引preflight；准备好的同目录可以随后去掉PREPARE_ONLY运行。即使准备阶段也会读取并hash模型权重，暂时安静不等于卡住，日志位于两个子目录`prepare.log`。常规运行的后续preflight还会读hash模型和完整索引，只有通过后才开始BM25。

## 新终端看进度

```bash
BASE="/local_data/$USER/research-agent-rl-data"
COMPARE_ROOT="$(cat "$BASE/runtime_efficiency/latest_retriever_compare.txt")"
echo "$COMPARE_ROOT"
ps -fp "$(cat "$COMPARE_ROOT/pid.txt")" || true
tail -n 20 "$COMPARE_ROOT/driver.log"
for NAME in bm25 qwen_hybrid; do
  echo "===== $NAME ====="
  tail -n 8 "$COMPARE_ROOT/$NAME/driver.log" 2>/dev/null || true
done
```

`[i/192]`是job在预生成顺序中的位置；resume时跳过完成job，不能把日志行数当作实际完成数。完成数查看各`summary.json`的arms.completed_episodes。

完成标志：顶层`RETRIEVER_COMPARE_COMPLETED` + `comparison.json.complete=true`。两个子run仍各自有完整的`manifest.json`、`summary.json`、`jobs/`、`cost.jsonl`、Qwen的`embedding_cost.jsonl`、服务器身份及预算invocation记录。

```bash
BASE="/local_data/$USER/research-agent-rl-data"
COMPARE_ROOT="$(cat "$BASE/runtime_efficiency/latest_retriever_compare.txt")"
cat "$COMPARE_ROOT/comparison.md"
```

运行中也可执行`python scripts/summarize_musique_retrievers.py --root <COMPARE_ROOT>`生成暂定比较；缺失或infrastructure_error/actor_error的job仅在配对中排除，不计成答错。脚本校验同job对应的task/种子身份，输出缺失清单。未complete时不能把子集均值当成完整结果。

## 指标与选择规则

主质量指标看同arm的Answer EM/normalized F1、grounded EM、citation F1、retrieved/read gold recall；原legacy task_success仍可在子runsummary查看，但不替代严格EM。成本看policy prompt/completion tokens、SEARCH/READ次数、episode_wall_sec及embedding费用。episode墙钟包含检索、工具执行与生成，而不是只有模型latency。

`comparison.json`对相同task/repeat/arm配对，并先在每道题内平均repeat差值再按task平均；报告paired_tasks，不将48次episode当48个独立任务。它是描述统计，不提供置信区间或显著性；该24题均衡hop探索样本不能称为自然分布held-out成绩。

成本汇总包含失败/重跑尝试。未知policy completion使用reservation上限保守计账；未知embedding usage保持unknown标识，不能报告为免费。若缺少成本日志，ledger_present=false；不能将reported token计数解释成完整费用。

优先验证adaptive_full是否解除HotpotQA四步模板的提前停止，结合其余三组分离prompt/READ效应；不盲选训练loss或reward高的一组。若Qwen只是增加候选召回而Agent仍一次SEARCH/READ即回答，应继续研究后续query与证据选择，不急着加step penalty。当前已知的candidate provenance与进展判定缺陷仍存在，两检索器代码保持一致以免混入额外改动。

本轮选定配置后，先锁协议和检索器，再建独立dev索引做更大held-out确认。后续RL比较冻结同一检索器、SFT初始化和采样预算；不把环境升级归因到算法创新。

## 下载回本机

在35拿到实际COMPARE_ROOT后，在Mac用实际目录名替换：

```bash
LOCAL="/Users/wishfine/Documents/RL/RL_project/ResearchAgent-RL/artifacts/runtime_efficiency/<run目录名>"
mkdir -p "$LOCAL"
rsync -avh xdf-35:/local_data/zhangyonglin/research-agent-rl-data/runtime_efficiency/<run目录名>/ "$LOCAL/"
```

这是文本轨迹、汇总和成本，不包含权重或索引。原始job/cost保留在本机，不纳Git；之后发布紧凑统计、协议manifest和真实结论。没有下载结果前不填模型提升。

## 本机验证与限制

全量pytest：137 passed、2 skipped；受影响的检索/协议/新比较测试40 passed。验证了配对任务/种子/模型/语料错配拒绝、基础设施失败排除、未知policy/embedding请求计账、失败embedding延迟保留、两个独立root的真实prepare流程、真实toy HTTP的无POST preflight及损坏索引拒绝。CLI检查覆盖JSON/Markdown产物和未完整时exit2，shell语法和diff检查通过。

独立只读审查指出并修复了子目录软链命名冲突及过晚检查Qwen readiness两项问题；复审无剩余Important/Critical问题。核心protocol客户端未修改。测试fixture不是实验模型结果；本机无35的GPU、真实权重或服务，真实384 episode仍需用户执行。macOS本机未实际执行依赖Linux `flock` 的完整shell协调器；它通过语法检查，Python客户端与preflight/汇总行为已单独验证。
