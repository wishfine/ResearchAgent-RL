# MuSiQue 冻结策略协议适配：实现与35运行说明

日期：2026-09-30。状态：代码与本机测试完成，**35上的新四组模型实验尚未运行**。

## 为什么先做这个

已下载的 `musique_sft874_infra_smoke10_20260930_134311` 是基础设施 smoke，不是性能基准：10题都只SEARCH一次、READ一次且一次选两段；没有继续检索，7次非法动作都是ANSWER先于CITE。最大单次输入+输出2363 tokens，不能归因于32K上下文不足。

这些行为与旧四步SFT模板一致，但没有受控比较，不能断言是SFT训练造成。READ只返回300字符也是设计风险，不过这10题不能证明截断是主因。先分别测试两种协议因素，不改检索排序、不训练模型、不预设提升。

## 四组对照

| arm | 提示词 | READ |
|---|---|---|
| cite_first_head300 | 现有多跳prompt + 明确CITE前置条件 | 前300字符，旧格式 |
| cite_first_full | 同上 | 完整段落 |
| adaptive_head300 | 同上 + 根据缺失关系继续检索、选择阅读 | 前300字符 |
| adaptive_full | 同上 | 完整段落 |

四组都要求每次只输出一个action，不展示推理；SEARCH→READ访问边界、READ→CITE→ANSWER前置条件不放宽。不强制至少搜索两次，不规定固定READ数量，不给模型gold分解/中间答案。自适应提示只允许使用原问题和真实历史工具结果中的实体或事实。

这个设计估计READ模式及额外多跳指导的影响；**不单独估计CITE提示的效应**。旧10题不是同题同seed对照，不能直接当第五个arm计算因果提升。

从MuSiQue **train** 固定抽24题：2/3/4-hop各8，`selection_seed=20260929`，每组每题2次独立请求seed，共192条episode。作业顺序固定随机打散，串行调用，防止四组同时挤压服务。均衡hop宏平均不同于自然分布的benchmark平均。

这是现有runtime-efficiency计划前的**协议适配诊断**。没有实现任意前缀快照/KEEP_A/REPLAN/ALT因果干预，不冒充完整P1。官方dev不用于本轮反复调试。

## 改动边界

- `ReadTool(mode="full")`保留全文；默认仍为`head300`，不重写旧结果。全文重读时同步刷新state中的summary。
- 修复自定义system_prompt被`multi_hop=True`覆盖的问题；未传自定义prompt时两个旧默认prompt不变。
- 客户端新增请求seed、usage是否真实提供、finish_reason；旧CLI和Actor调用仍兼容。
- SEARCH只新增执行边界参数检查：非空字符串query，整数topk∈[1,100]（bool不算int）。新实验四组统一使用这个协议。
- 旧JSON BM25不再用score>0判断匹配，改为查询词实际命中；处理common-term负IDF，并固定并列顺序。**SQLite FTS5排名、title/content权重、snippet都未变。**旧历史JSON结果不追溯覆盖。

## 结果、成本与续跑

驱动：`scripts/run_musique_protocol_smoke.py`；35封装：`scripts/run_musique_protocol_smoke.sh`。

- `manifest.json`：抽题及task文件SHA、语料SHA、实际源码SHA、Git commit、prompt全文、所有参数与作业seed；给`--model_dir`时包括JSON/模板/权重文件SHA。首次扫描模型18GB可能需要一些时间，不是卡在推理。
- `server_models.json`：核验served ID、root及32K服务配置。要求独立冻结服务，中途不得reload权重；路径核验不等于远端运行中权重的密码学证明。
- `jobs/*.json`：每条轨迹、原始回复、工具结果、usage/finish_reason、模型调用累计延迟和episode端到端wall time。
- `summary.json`：四组覆盖情况、按hop EM、paired EM差（对cite_first_head300）、旧指标与新增诊断分开存。重复采样不是独立题目，不声称小样本统计显著或算法有效。
- 新诊断：自定义normalized EM/F1（含aliases）、grounded EM、检索/阅读gold recall、SEARCH次数和query多样性、length结束次数。**不是官方MuSiQue evaluator**。
- grounded EM要求精确答案、ANSWER成功、包含全部gold支持段落，且最终引用均已READ/CITE。不验证每条claim的语义蕴含；源标注疑点不自动删除或改gold。
- `cost.jsonl`：请求前fsync写预算预留，请求后记录实际usage；缺失usage、失败或进程中断保留max_tokens预留，不能当免费。包括失败和重跑尝试，不只计完成样本。
- 默认累计2880 API请求、1,474,560 completion token上限；wall cap每次启动4小时，达到预算停止并保留未完成状态。BM25单次在途生成最多可额外等待客户端60秒timeout；embedding模式工具请求timeout为120秒，wall检查可能再延后。wall预算不是跨重启累计预算，call/token预算是跨续跑累计的。
- 同一个RUN_DIR、同一配置重新执行，跳过已完成episode；有基础设施/预算中断的episode从头重跑（**不是任意turn恢复**），旧失败尝试另存`.attempt_*.json`，成本保留。可增加预算，但不能改变抽题/模型/采样/代码；改变需新RUN_DIR。
- 文件锁阻止两个驱动同时写同一目录。ledger写坏或manifest不一致会停止，不偷偷修复/清空。
- `complete=false`时，现有均值只是部分结果，不能做公平组间结论。预算中断不计模型失败或0分；正常invalid/max_steps终止计作完整episode。

## 在35执行：使用现有GPU0服务，不额外占卡

先确认8105仍是SFT874且32K；此脚本不启停vLLM、不执行`ray stop`、不影响别的项目。

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
PY="/local_data/$USER/conda_envs/research-agent-runtime/bin/python"
cd "$PROJECT"
git pull --ff-only origin slime-rewrite

curl -fsS --max-time 10 http://127.0.0.1:8105/v1/models | "$PY" -m json.tool

RUN_DIR="$BASE/runtime_efficiency/musique_protocol24_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
PROJECT="$PROJECT" BASE="$BASE" PY="$PY" RUN_DIR="$RUN_DIR" \
  nohup bash scripts/run_musique_protocol_smoke.sh \
  >"$RUN_DIR/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/pid.txt"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$PID"
)
```

默认保存到`/local_data/$USER/research-agent-rl-data/runtime_efficiency`，自动在本项目`artifacts/runtime_efficiency`下建对应软链。生成的协议运行目录已gitignore；不把权重/数据集/所有轨迹提交Git。

新终端查看（不依赖此前shell变量）：

```bash
BASE="/local_data/$USER/research-agent-rl-data"
PY="/local_data/$USER/conda_envs/research-agent-runtime/bin/python"
RUN_DIR="$(cat "$BASE/runtime_efficiency/latest_protocol_smoke.txt")"
test -n "$RUN_DIR" && test -d "$RUN_DIR"
echo "$RUN_DIR"
ps -fp "$(cat "$RUN_DIR/pid.txt")" 2>/dev/null || true
tail -n 40 "$RUN_DIR/driver.log"
test ! -f "$RUN_DIR/summary.json" || "$PY" -m json.tool "$RUN_DIR/summary.json"
```

预算或网络中断后手动续跑：先确认原PID已结束，再用同一个RUN_DIR执行上述nohup行；不要重建timestamp目录，也不要只降低repeats。若需初步96条，可第一次启动前设`REPEATS=1`，另建实验目录，不与2次重复结果混用。

结束后在本机同步（客户端目录已存在则保留其其他内容，不用--delete）：

```bash
cd /Users/wishfine/Documents/RL/RL_project/ResearchAgent-RL
REMOTE_RUN="$(ssh xdf-35 'cat /local_data/zhangyonglin/research-agent-rl-data/runtime_efficiency/latest_protocol_smoke.txt')"
test -n "$REMOTE_RUN"
LOCAL_RUN="$PWD/artifacts/runtime_efficiency/$(basename "$REMOTE_RUN")"
mkdir -p "$LOCAL_RUN"
rsync -avh --include='jobs/' --include='jobs/*.json' \
  --include='manifest.json' --include='summary.json' --include='cost.jsonl' \
  --include='server_models.json' --include='invocation_*.json' \
  --include='driver.log' --exclude='*' "xdf-35:$REMOTE_RUN/" "$LOCAL_RUN/"
```

## 本机验证

测试先失败再修复，覆盖全文READ、访问边界、prompt覆盖、SEARCH坏参数、分层抽样、不泄漏gold、usage/seed、写前成本预留、真实HTTP toy策略四组链路、预算中断、续跑跳过/保留尝试和manifest拒绝混写。toy策略测试不是模型实验结果。

2026-09-30：全量pytest **95 passed / 2 skipped**（新增测试与旧JSON bug修复后）；shell语法通过。真实数据24题manifest准备通过；GPU模型实验待用户35运行。

后续新增可选dense/hybrid接入，默认仍BM25；新run记录索引完整manifest、encoder指纹及独立embedding成本日志。embedding故障记infrastructure_error并排除公平性能汇总。BGE/Qwen服务、token窗口索引与召回对照命令见`embedding_retrieval_runbook_20260930.md`。不更改旧run的检索模式。

最后一轮包含embedding功能的全量验证为106 passed、2 skipped；95项通过是此前仅协议阶段的历史结果。
