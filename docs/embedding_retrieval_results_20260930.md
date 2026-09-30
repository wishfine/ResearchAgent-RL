# MuSiQue：BGE / Qwen 五组离线检索结果

日期：2026-09-30。性质：**train discovery 检索器对照，不是 Agent 评测或 RL 增益**。

## 证据与协议

- 服务器35 run：`/local_data/zhangyonglin/research-agent-rl-data/runtime_efficiency/embedding_retrieval_audit_20260930_165424`。
- 用户下载原件：`/Users/wishfine/Documents/RL/RL_project/embedding_retrieval_audit_20260930_165424`。已完整复制到本项目 `artifacts/runtime_efficiency/embedding_retrieval_audit_20260930_165424/`，包括请求成本日志；不纳入 Git。
- 可版本化的五组原始汇总、完成日志与 commit：`artifacts/runtime_efficiency/search_audit_20260930/embedding_165424/`。
- 运行 commit：`044a57639b9ddc9e028209149eead225d6ecef04`。
- 按该 commit 的 launcher：train 固定 seed42、1000题、Top-5/10/20；每组均核对 `tasks=1000`、`corpus_chunks=83866`，hop 分布为730/207/63道2/3/4-hop。
- `driver.log` 含 `RETRIEVAL_AUDIT_COMPLETED`，四个 embedding 请求日志均有3333个 start、3333个 success，无 error、无未结束请求；所有成功请求均记录真实 usage 与 canonical 输入策略。
- 五组使用原问题检索，并额外测1000题的2333个 gold hop 查询。Oracle 桥接查询含正确中间答案，**仅作离线诊断，不进入索引、Agent 提示或实际策略输入**。
- 汇总没有保存逐题排名/命中向量及完整任务文件清单，因此本次不能据此计算 paired bootstrap、McNemar 检验或逐题回归分析。相同选样程序与 seed、题数和 hop 分布支持协议一致性，但不等于完整逐题数据版本证明。

## 1. 原问题单次搜索

`support_recall` 为每题支持段落召回的宏平均；`all_support_recall` 为全部标注支持段落被召回的题目比例。不是回答正确率。

| 方法 | TopK | 支持段落召回 | 全支持段落命中 | 首跳命中 |
|---|---:|---:|---:|---:|
| BM25 | 5 | 48.75% | 13.40% | 78.60% |
| BGE dense | 5 | 2.31% | 0.00% | 1.50% |
| BGE hybrid | 5 | 32.57% | 4.70% | 52.90% |
| Qwen dense | 5 | 50.61% | 18.10% | 73.10% |
| Qwen hybrid | 5 | **54.27%** | **20.90%** | **80.10%** |
| BM25 | 10 | 55.10% | 20.50% | 82.90% |
| BGE dense | 10 | 3.52% | 0.00% | 1.80% |
| BGE hybrid | 10 | 44.37% | 12.10% | 69.50% |
| Qwen dense | 10 | 57.76% | 26.60% | 79.10% |
| Qwen hybrid | 10 | **61.03%** | **29.10%** | **84.90%** |
| BM25 | 20 | 59.26% | 26.20% | 86.30% |
| BGE dense | 20 | 5.13% | 0.00% | 2.90% |
| BGE hybrid | 20 | 55.66% | 21.70% | 81.90% |
| Qwen dense | 20 | 64.17% | 34.90% | 83.50% |
| Qwen hybrid | 20 | **67.42%** | **37.90%** | **88.80%** |

表格百分比由原始浮点值展示舍入，不作为计算源；精确值保留在 JSON。

Qwen hybrid 相对 BM25 在 Top-10 的支持段落召回增加5.93个百分点、全支持段落命中增加8.60个百分点（205→291题）、首跳命中增加2.00个百分点。Top-20 全证据命中增加11.70个百分点（262→379题）。这些是检索结果差值，不是显著性检验或 RL 提升。

BGE dense 的 Top-10 支持段落召回仅3.525%，不是可用的英文 MuSiQue 主检索器；等权 RRF 接入它后，Top-10 全证据命中从 BM25 的20.5%下降到12.1%。**融合不保证提高质量**，不能默认任何 embedding 通道都应加入。

此前 HF/vLLM 输出一致性已在35验证，当前索引加载也校验 corpus/model/tokenizer/向量身份。BGE 的中文模型与英文任务不匹配是合理假设，但这组数据不能证明语言因素是唯一原因；若专门追查，还需要独立检查 query/document 排名样例、英文编码能力和其他 BGE 配置。首轮为空的 query instruction 是冻结配置，不应把本结果推广成“所有 BGE 都差”。

## 2. Oracle 多跳检索诊断

| 方法 | 每跳召回@10 | 完整链命中@5 | 完整链命中@10 | 完整链命中@20 |
|---|---:|---:|---:|---:|
| BM25 | 82.43% | 55.30% | 65.10% | 73.20% |
| BGE dense | 6.73% | 0.30% | 0.40% | 0.40% |
| BGE hybrid | 72.27% | 29.50% | 47.90% | 65.40% |
| Qwen dense | **87.31%** | **66.20%** | **73.20%** | 77.80% |
| Qwen hybrid | 87.14% | 64.40% | 72.50% | **82.40%** |

Qwen hybrid 并非每一项都最好：更短、更具体的 gold hop 查询在@5/@10上，Qwen dense 的完整链命中略高于 hybrid。后续保留 dense 为消融对照；不能仅凭原问题结果宣布 hybrid 对所有 query 都优。

同一 Qwen hybrid 的原问题全证据@10为29.1%，gold hop 完整链@10为72.5%，反映两种查询条件下仍有较大差距。**它不是可保证的 Agent 提升空间，也不是 query planning 的因果效应**：查询数、分解和答案信息均不同。只能用于提出“学习后续桥接 query 是否有价值”的实验假设。

## 3. 在线 query 编码成本

每组3333个单条请求（原问题1000 + gold hops2333）；不包含建库、服务器启动和 Agent 生成成本。P50/P95 使用所有成功请求的线性插值分位数，保留各组首个请求，不删除高延迟样本。

| 方法 | embedding prompt tokens | 客户端请求P50 | P95 | 最大请求 | 请求延迟总和 | 首个start→最后success |
|---|---:|---:|---:|---:|---:|---:|
| BGE dense | 58,673 | 18.70 ms | 20.39 ms | 3.63 s | 63.50 s | 428.19 s |
| BGE hybrid | 58,673 | 18.78 ms | 20.33 ms | 3.29 s | 62.14 s | 505.20 s |
| Qwen dense | 107,441 | 42.82 ms | 45.81 ms | 4.85 s | 147.15 s | 573.48 s |
| Qwen hybrid | 107,441 | 42.79 ms | 45.59 ms | 4.21 s | 144.39 s | 658.00 s |

计时位于客户端 `EmbeddingClient.encode`，包括本地 tokenization、HTTP 往返、服务编码、解析/归一化及部分日志处理，**不是纯 GPU kernel 时间**。请求后执行的 CPU 向量检索、RRF 和其他诊断工作不在单请求 latency 中；因此45.59 ms不能当作 SEARCH 工具P95。四组日志时间跨度也不等于精确进程总运行时间，BM25 没有同类请求成本日志。

Qwen query tokens 较多包含模型 tokenizer 和 query instruction 差异，不能把模型间的原始token数量直接视为等价算力单位。完整检索audit不是高并发压测；没有 GPU 使用率、共享卡实际部署日志或端到端 Agent 耗时，不能证明高并发RL性能。用户要求两个embedding服务共享GPU1，但本次下载物本身不足以审计实际GPU分配。

## 4. 索引身份

两种模式中同一模型的 index manifest 一致，均为 `dense-windows-v2`、`complete=true`、`local_hf_token_ids_v1`、窗口重叠64、段落分数取最大窗口余弦；语料SHA为 `dd41ddf18f21f7085ac7a6ca7c3858f634f38c0ce1247d09a07ca2083ac80181`。

| 模型 | pooling | 维数 | 窗口cap | 段落数 | 窗口行数 |
|---|---|---:|---:|---:|---:|
| BGE small zh v1.5 | CLS | 512 | 512 | 83,866 | 83,996 |
| Qwen3 Embedding 0.6B | LAST | 1024 | 2048 | 83,866 | 83,866 |

BGE 的额外130行是额外窗口，不是130个新段落，也不能推断恰有130个长段落。窗口策略、instruction和模型均不同，本实验对比的是完整检索配置，不隔离单一模型参数量效应。

| 原始结果文件 | SHA-256 |
|---|---|
| bm25.json | `e1a5cb0783be3f5fa7a91085ca10a1787c19d1b99c0add734b9fc2e94e895f9a` |
| bge_dense.json | `a51068fe69c771ae5d1001d9e701bdb19778a658683882dd33a79edcd122a02e` |
| bge_hybrid.json | `101b366e5d82f79553b5a39c4091c47dd3d8bfe8736b3fe870f598477ed864ed` |
| qwen_dense.json | `6713b0c883db97af9039163bca7fe24d019209471f41ff81d1ae5289b06b93f8` |
| qwen_hybrid.json | `1b4531ecb3cda56e03775e112ada3bc9a6b1c239c0d57466850671642db7f561` |

## 5. 下一步决策

1. 以 Qwen hybrid 作为下一阶段主检索候选，保留 BM25 与 Qwen dense 对照；BGE 中文配置保留负结果，不用于当前英文主实验。
2. 用冻结 SFT874 在相同24道train任务、相同采样种子和重复数、prompt/READ四组条件下，先比较 BM25 与 Qwen hybrid。原问题单次召回较高不能代替实际Agent验证。
3. 看 Answer EM/token F1、有效引用、未召回/未选读/提前停止、后续 query、输入/输出token及端到端延迟，同时计入embedding成本。不要因检索较好就自动提高topk或把所有候选全读。
4. 本次是 train discovery，不做 held-out 效果宣传。协议与检索器锁定后，另建 eval 索引并在dev上确认；RL基线与新算法使用同一个冻结检索器，不混淆环境升级和算法增益。

检索基础设施已完成，但新Agent对照和RL算法有效性尚未验证。候选provenance陈旧、length-only进展判定、query-centered snippet及学习式reranker仍是独立待办。
