# BGE / Qwen embedding：全库检索对照与35运行命令

日期：2026-09-30。实现完成，本机toy/HTTP/状态回归测试通过；**实际embedding模型GPU服务、全量建索引及性能结果仍待35执行验证**。没有替用户启动远端服务，没有下载模型到本机，没有修改其他项目环境。

## 做了什么

SEARCH保留原动作协议，三个检索模式：

- `bm25`：现有SQLite FTS5，不调用embedding。
- `dense`：query embedding与所有段落窗口embedding做精确cosine相似度检索。
- `hybrid`：分别取lexical/dense候选，按等权RRF融合，`rrf_k=60`，每通道至少50个候选再选最终top-k。BM25原始分数和cosine不可直接相加；RRF不是创新算法。

无需FAISS/FlagEmbedding/sentence-transformers新增依赖；建索引用已有Transformers tokenizer，编码复用vLLM `/v1/embeddings`。本轮按83,866段做CPU精确矩阵检索，目标是可控baseline，不宣称已达到高并发RL rollout性能；之后按实测p95决定是否换ANN或集中检索服务。

两种模型均支持、索引不混用：

| 模型 | pooling | 最大窗口token（含特殊token） | query instruction |
|---|---|---:|---|
| 本地 `bge-small-zh-v1.5` | CLS | 512 | 首轮为空，冻结配置 |
| `Qwen3-Embedding-0.6B` | LAST | 2048（服务cap4096） | 官方通用检索instruction，仅加在query |

BGE模型卡将该模型列为中文模型；MuSiQue是英文，因此这只是无需下载的对照，不默认它优于BM25。其config hidden_size=512，max_position_embeddings=512。[BGE模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[config](https://huggingface.co/BAAI/bge-small-zh-v1.5/raw/main/config.json)。

Qwen0.6B支持1024维、32K，官方示例采用LAST pooling和归一化，检索query带instruction、document不加instruction。[Qwen模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。当前vLLM0.23用`--runner pooling --convert embed`；pooler参数用`use_activation`，**不是旧版normalize键**。[vLLM0.23 pooling说明](https://docs.vllm.ai/en/v0.23.0/models/pooling_models/)、[PoolerConfig源码](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/config/pooler.py)。

## 索引与证据边界

编码内容是`title + newline + paragraph content`，没有问题答案、ground_truth_citations或gold分解。train/eval必须分开建索引，corpus SHA不一致直接拒绝加载。

长段落按本地tokenizer切重叠64-token窗口，不只取前512；窗口直接发送token IDs，保留实际特殊token。段落分数是所有窗口cosine的最大值；最终返回**原paragraph/chunk ID**，没有把窗口伪装成新的证据。READ仍读原段落，支持原head300/新增full模式。不同模型的窗口策略本身也是检索配置的一部分，不能把二者差异全归因于模型参数量。

索引包含`build.json`、`progress.json`（每批checksum）、`rows.json`、float32 `vectors.npy`和完成后的`manifest.json`。可中断后续建；校验已提交批次、完整文件SHA、语料/model/tokenizer/pooling/instruction身份后才续跑。不同源、损坏索引或encoder变更必须另建目录，不覆盖旧索引。模型已在同一路径被替换时，启动会重新比对实际文件SHA而不是只相信索引里复制出来的fingerprint。

模型服务仍必须专用冻结：`/v1/models`的ID/root/长度和本地文件hash不能证明服务运行中权重从未被外部reload。不要中途换权重、tokenizer或pooling。建索引和加载时读模型SHA会花时间。

每次embedding请求写`embedding_cost.jsonl`：调用前记录请求、batch数量、实际输入hash，成功记录真实usage/延迟，失败记error。失败/中断不算免费。没有真实usage就标记未知，不补0作为真实成本。embedding服务断连在Agent中记`infrastructure_error`并停止，不计成policy非法动作。

## 1. 35拉取代码，按需下载Qwen

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
PY="/local_data/$USER/conda_envs/research-agent-runtime/bin/python"
cd "$PROJECT"
git pull --ff-only origin slime-rewrite
"$PY" scripts/download_qwen_embedding.py \
  --model_dir "/local_data/$USER/models/Qwen3-Embedding-0.6B"
)
```

下载脚本解析并固定实际Hub revision，保存receipt；网络断了可重复同条命令按receipt继续。已存在无receipt的模型目录不覆盖，请确认后复用或另选路径。BGE本地模型不下载。

## 2. 启动两个embedding服务

现有SFT模型服务仍在GPU0:8105，不启停它。以下默认检查空闲GPU1和GPU2；被占用就拒绝启动，可以修改`EMBED_GPU`，不杀别人进程。每个embedding服务GPU预算比例0.10，在A80080GiB上是约8GiB引擎预算，**不是实测显存使用**。

```bash
cd /local_data/$USER/ResearchAgent-RL
EMBED_GPU=1 EMBED_PORT=8106 bash scripts/start_embedding_server.sh bge
EMBED_GPU=2 EMBED_PORT=8107 bash scripts/start_embedding_server.sh qwen
```

BGE默认路径即用户已有的`/local_data/zhangyonglin/data/bio-know-tag/models/bge-small-zh-v1.5`；Qwen用第1步路径。启动脚本offline加载，不自动pip install或联网找权重。

查看启动日志，不依赖之前shell变量：

```bash
BASE="/local_data/$USER/research-agent-rl-data"
for FAMILY in bge qwen; do
  SERVER_DIR="$(cat "$BASE/runtime_efficiency/latest_embedding_${FAMILY}.txt")"
  echo "$SERVER_DIR"
  ps -fp "$(cat "$SERVER_DIR/pid.txt")" 2>/dev/null || true
  tail -n 40 "$SERVER_DIR/server.log"
done
curl -fsS --max-time 10 http://127.0.0.1:8106/v1/models
curl -fsS --max-time 10 http://127.0.0.1:8107/v1/models
```

两端/models返回后，**先验证pooling及文本/token-ID编码一致性**；CPU加载reference只跑3条短文本，不额外占GPU：

```bash
(
set -euo pipefail
cd /local_data/$USER/ResearchAgent-RL
PY="/local_data/$USER/conda_envs/research-agent-runtime/bin/python"
OMP_NUM_THREADS=4 "$PY" scripts/verify_embedding_backend.py \
  --model_dir "/local_data/$USER/data/bio-know-tag/models/bge-small-zh-v1.5" \
  --embedding_url http://127.0.0.1:8106/v1 --embedding_model bge-small-zh-v1.5
OMP_NUM_THREADS=4 "$PY" scripts/verify_embedding_backend.py \
  --model_dir "/local_data/$USER/models/Qwen3-Embedding-0.6B" \
  --embedding_url http://127.0.0.1:8107/v1 --embedding_model Qwen3-Embedding-0.6B
)
```

若验证不通过，保留server.log并修复，不跳过它跑“效果实验”。这是backend正确性验证，不是模型质量验证。

## 3. 后台建train两个索引（可续建）

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
PY="/local_data/$USER/conda_envs/research-agent-runtime/bin/python"
cd "$PROJECT"
mkdir -p "$BASE/dense_indexes" "$BASE/runtime_efficiency"
BUILD_DIR="$BASE/runtime_efficiency/embedding_build_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BUILD_DIR"

nohup "$PY" -u scripts/build_dense_index.py \
  --corpus_dir "$PROJECT/data/musique_rl_v2/corpus/train" \
  --index_dir "$BASE/dense_indexes/musique_train_bge_small_zh_v15" \
  --model_dir "/local_data/$USER/data/bio-know-tag/models/bge-small-zh-v1.5" \
  --embedding_url http://127.0.0.1:8106/v1 --embedding_model bge-small-zh-v1.5 \
  --pooling CLS --max_length 512 --overlap 64 --batch_size 32 \
  >"$BUILD_DIR/bge.log" 2>&1 < /dev/null &
echo $! > "$BUILD_DIR/bge.pid"

nohup "$PY" -u scripts/build_dense_index.py \
  --corpus_dir "$PROJECT/data/musique_rl_v2/corpus/train" \
  --index_dir "$BASE/dense_indexes/musique_train_qwen3_embed_06b" \
  --model_dir "/local_data/$USER/models/Qwen3-Embedding-0.6B" \
  --embedding_url http://127.0.0.1:8107/v1 --embedding_model Qwen3-Embedding-0.6B \
  --pooling LAST --max_length 2048 --overlap 64 --batch_size 32 \
  >"$BUILD_DIR/qwen.log" 2>&1 < /dev/null &
echo $! > "$BUILD_DIR/qwen.pid"
printf '%s\n' "$BUILD_DIR" > "$BASE/runtime_efficiency/latest_embedding_build.txt"
echo "BUILD_DIR=$BUILD_DIR"
)
```

完成标志是两个index_dir的`manifest.json`中complete=true，不是只存在vectors.npy。重跑同一建索引命令会核验并继续，不同时起两个同目录driver。损坏时另建目录，别删除日志掩盖原因。

按没有额外窗口粗算：train float32向量 BGE512维约164MiB、Qwen1024维约328MiB；有窗口会增加，另有row metadata/CPU工作空间。模型权重/GPU激活不在这个数字里，速度和显存以实测为准。

## 4. 固定1000题做五组检索对照

索引完整后：

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
cd "$PROJECT"
RUN_DIR="$BASE/runtime_efficiency/embedding_retrieval_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
RUN_DIR="$RUN_DIR" AUDIT_TASKS=1000 \
  nohup bash scripts/run_embedding_retrieval_audit.sh \
  >"$RUN_DIR/driver.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid.txt"
printf '%s\n' "$RUN_DIR" > "$BASE/runtime_efficiency/latest_embedding_audit.txt"
echo "RUN_DIR=$RUN_DIR"
)
```

五组是BM25、BGE dense/hybrid、Qwen dense/hybrid，完全相同train1000题seed42与topk5/10/20。输出各组JSON/log/embedding成本日志。除了原问题检索，该诊断也会用gold桥接query测oracle召回；**oracle仅用于离线诊断，不进入Agent提示或索引**。默认不是只有1000个embedding请求，gold hop诊断还会增加请求；依日志统计实际数量。

可以首次把`AUDIT_TASKS=100`用于基础设施检查，但要另建结果目录，不能混入1000题表。如果只启了Qwen，设`EMBED_AUDIT_MODELS=qwen`，得到BM25+Qwen两组检索模式。

## 5. 接入Agent：同题同参数，另建run目录

原四组prompt/READ实验默认BM25。embedding版本必须另建目录，不中途换旧run的检索器。以下用Qwen hybrid运行同一24题2次重复；保留对应BM25四组作为对照：

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
cd "$PROJECT"
RUN_DIR="$BASE/runtime_efficiency/musique_protocol24_qwen_hybrid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
RETRIEVAL=hybrid \
DENSE_INDEX="$BASE/dense_indexes/musique_train_qwen3_embed_06b" \
EMBEDDING_URL=http://127.0.0.1:8107/v1 \
RUN_DIR="$RUN_DIR" \
nohup bash scripts/run_musique_protocol_smoke.sh \
  >"$RUN_DIR/driver.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid.txt"
echo "RUN_DIR=$RUN_DIR"
)
```

改`RETRIEVAL=dense`测试dense-only；BGE则修改DENSE_INDEX为BGE索引、EMBEDDING_URL为8106。不要在同一个output_dir中换这三项。无需重新部署SFT模型，也不需要训练。

生成token/API-call预算仍是policy预算；embedding的建库和在线query成本独立记录，比较效率时必须合计，不能称为“免费工具”。embedding120秒timeout可能让wall预算检查晚于上限；新请求前检查预算，单个在途embedding请求最多额外等待120秒。完整任意前缀干预/P2并未随此实现完成。

## 6. 发回本机哪些文件

无需传权重和vectors.npy。传检索audit目录的JSON/log/embedding成本日志、两个索引manifest/build/progress、服务run_info/server.log、backend验证输出，以及Agent目录manifest/summary/jobs/cost/embedding_cost/server_models/invocation/driver.log。然后才能记录实际效果和耗时，当前不能预填提升。

模型选择依赖train discovery结果，协议锁定后再在held-out dev测；训练算法比较必须使用同一冻结检索器，不能把BM25→hybrid的增益算成RL增益。

## 验证记录

本机Python3.12全量测试：**106 passed、2 skipped**；三个shell脚本语法和Python CLI帮助入口通过。新增测试包含真实toy HTTP文本/token-ID请求、返回乱序恢复、向量归一化、窗口后文覆盖、原段落ID去重、docscope、RRF接口、损坏的完整/未完成索引拒绝、模型文件替换拒绝、infra_error与policy-invalid分离。独立只读审查发现的两项索引完整性问题均已修复并验证。

这些验证不等于BGE/Qwen实际GPU inference通过；第2步backend检查及后续真实召回/Agent结果仍待35执行。
