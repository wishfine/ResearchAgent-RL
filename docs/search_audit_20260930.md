# SEARCH审查：可用基线，尚非完善的多跳检索系统

日期：2026-09-30。只读实现审查 + 真实MuSiQue train离线检索复测；无模型调用。

## 现在究竟怎么搜

当前MuSiQue `retrieval_scope=split_corpus`、`reference_docs=[]`，对**该split全库**83,866段进行检索，不是只搜题目的20个候选；eval使用另一套隔离语料。查询由Actor生成，SEARCH工具不替Actor规划query。

SQLite FTS5流程：Unicode分词/小写 → 去英语停用词 → 去重 → 最多24词 → quoted OR匹配 → BM25(title权重5、content权重1) → top-k；并列按chunk ID固定排序。FTS参数绑定，不允许模型query作为SQL/FTS程序直接执行。标题全文参与排名，observation显示标题和正文**前200字符**，不是query-centered snippet。

旧HotpotQA的task_docs是不同协议：通常只在本题约10段中搜索；当topk足够大时补齐零分候选。旧高分不能拿来说明MuSiQue全库检索已经成熟。

## 复测结果

来源：`artifacts/runtime_efficiency/search_audit_20260930/retrieval_train_1000_seed42.json`。从train固定seed42抽1000题（730/207/63道2/3/4-hop），83,866段；与已有`data/musique_rl_v2/retrieval_train_smoke1000.json`一致。下表不是模型成功率。

| 单次用原问题搜索 | 支持段落macro recall | 全部支持段落命中比例 | 第一跳命中比例 |
|---|---:|---:|---:|
| top-5 | 48.75% | 13.4% | 78.6% |
| top-10 | 55.10% | 20.5% | 82.9% |
| top-20 | 59.26% | 26.2% | 86.3% |

在同批任务上，使用gold分解和gold桥接答案构造每一跳query时，top-10 oracle hop recall为82.43%，oracle完整链命中65.1%。**这是带答案的检索诊断，不是部署策略，不是Agent的可实现成功率。**它说明短而具体的后续query值得验证，但不能当作真实提升。

当前10题真实smoke全部query=原问题，没利用已读实体继续搜索。离线召回不足与策略不会展开多跳两者并存；不能把全部错误推给SEARCH，也不能认为只加dense retrieval就解决了。

## 哪些做好了，哪些没有

| 部分 | 判断 |
|---|---|
| 全split检索、train/eval隔离 | 已有；依赖数据转换和固定corpus版本 |
| 可重现排名、只读索引、query安全绑定 | SQLite路径已有 |
| READ只能访问已召回ID，CITE只能引用已读 | 已有能力边界，不靠模型自觉 |
| query/topk检查 | 本轮补齐非空字符串、整数1–100，拒绝bool/负数/超限 |
| 实体短语/词序、别名、语义匹配 | 没有；OR检索会丢词序，实体歧义明显 |
| hybrid/dense或cross-encoder排序 | 没有；RERANK仅snippet词重叠+已有score，不是学习式reranker |
| 关键词位置snippet、充分阅读 | snippet取正文头200；原READ头300，新全文模式可独立验证 |
| 新query后的候选状态刷新 | 候选按ID追加，重复ID的cached query/score/rank仍是首次记录；最新SEARCH结果有新排名，但state/RERANK有陈旧信息风险 |
| 进展判定 | 只看候选/阅读/引用列表长度；排序或同数量内容更新不算新进展，可能影响多轮探索 |
| JSON与SQLite行为一致 | 不一致：旧JSON按whitespace处理正文、title不参与BM25；本轮只修负/零分匹配丢失，不宣称两后端统一 |
| query规划 | Actor的职责；现有SFT策略本批未产生桥接query，不是SEARCH自动补全能代替的 |

## 本轮修什么、暂时不改什么

修了执行边界参数校验和旧JSON common-term负/零分被错误过滤；后者有失败回归测试并使已有HotpotQA准备测试恢复。JSON并列结果改为确定性ID顺序。SQLite排名保持不变，保证新协议四组对照不混入检索器变化。旧历史实验不覆盖重算。

候选状态刷新、语义progress、query-centered snippet和实体/hybrid排序本轮只审查、**尚未实现**。不能把当前版本称作这些功能都完成的“search v2”。

## 接下来升级的优先顺序

1. 先跑24题READ/prompt对照，分清未召回、未选读、读到后答错、提前停止；失败分析同时看源数据歧义，不凭结果删题。
2. 下一版优先修候选metadata/进展语义；明确排序、去重、query provenance契约，补多轮状态回归。作为单独协议版本，不能中途改本轮实验。
3. 增加query-centered snippet和实体/短语检索对照：短语召回+原OR通道融合，保留原通道避免只追精确实体而漏召回。测recall@5/10/20、完整链命中和p50/p95工具耗时。
4. 词法仍不足再测dense+lexical融合及小型reranker；同一任务/预算下比较Answer EM、citation recall、tool cost、累计输入/生成token与端到端时间，不只看召回。

不要直接把topk增到100让模型全部READ；那会增加提示输入和噪声。也不能凭目前短路径失败就给更强step penalty：先确认证据不足时能否合理多搜索。

复现：

```bash
python scripts/evaluate_retrieval.py \
  --tasks_dir data/musique_rl_v2/tasks/train \
  --corpus_dir data/musique_rl_v2/corpus/train \
  --topk 5 10 20 --max_tasks 1000 --seed 42 \
  --output_file artifacts/runtime_efficiency/search_audit_20260930/retrieval_train_1000_seed42.json
```

结论：**SEARCH的基础功能和访问边界可用，但召回能力、语义排序和多轮状态尚不完善。先把策略适配对照跑起来，再独立升级检索器。**

## 用户追加后实现的可选embedding通道

用户随后要求同时支持现有BGE和Qwen embedding。本轮已新增两个模型的独立索引、dense和RRF hybrid模式及Agent客户端接入；默认BM25不变。上表的“没有hybrid/dense”是审查原始代码时的事实，当前可选代码通道已存在。35的1000题真实检索对照现已完成：Qwen hybrid全证据@10为29.1%（BM25为20.5%），BGE hybrid为12.1%，因此优先验证Qwen hybrid，保留BM25对照。完整结果和客户端编码成本见`embedding_retrieval_results_20260930.md`；仍未测新检索器的Agent效果及完整工具p95。没有加入cross-encoder reranker，也没有修复候选provenance或length-only进展判定；不能将这些待做项也称为已完成。运行方式见`embedding_retrieval_runbook_20260930.md`。
