# 45 → 35：实验环境准备

日期：2026-09-30。目标：在 172.22.0.35 的 A800 上运行冻结策略的 MuSiQue 诊断。

## 当前证据

- 本机 SSH 配置：`xdf-35` / `xdf-45`，均为用户 `zhangyonglin`，22 端口。
- 本机测试两台机器 TCP 22 均可达；SSH 在密钥交换前返回 `Connection closed`，尚未登录。不能据此判定密码错误、环境存在或磁盘共享。
- 用户报告 35 有 8 张空闲 A800；正式启动前重新核验 GPU 进程。
- 用户最初认为 `/data` 共享；随后两端实际挂载输出表明：45的环境/模型在本机XFS `/data`（UUID `5913bdc1-9770-41ba-9de7-2c581d031a77`），35的 `/data/zhangyonglin` 不存在，`/data` 未挂载该盘。当前不能直接复用原绝对路径。
- 35可用存储：本机ext4 `/local_data` 约20T可用；根分区约14G、`/home` 约46G可用。35的两个NFS挂载来自172.22.0.38，不是已验证的45数据导出。
- 35已有环境：`/local_data/zhangyonglin/conda_envs/english-kp-qwen35`、`bio-know-tag-dense`、home下`agentgym`。先盘点推理依赖，避免不必要的完整训练环境迁移；不修改另一个项目的环境。
- 35驱动535.183.01，8张A80080GB在用户盘点时各13MiB占用。实际启动前仍需检查进程；torch CUDA兼容性需实际kernel验证。
- 历史环境线索：`/data/zhangyonglin/conda_envs/vime-train-cu129`，torch 2.11.0+cu129、vLLM 0.23.0+cu129；本轮实际版本待盘点。
- 历史数据线索：`/data/zhangyonglin/research-agent-rl-data/musique_rl_v2`。本机已有同名数据目录约 203 MiB。
- 最新 runtime-efficiency 计划中的新采集/分支脚本在本机尚不存在；已有 effect-space pilot 是不同诊断，不能冒充完整新实验。

## 1. 先收集环境盘点

以下在本机项目目录执行，只读服务器信息。每次使用新日志目录；若 SSH 失败，日志不是环境盘点成功证据。

```bash
cd /Users/wishfine/Documents/RL/RL_project/ResearchAgent-RL
set -o pipefail
PROBE_OUT="$PWD/artifacts/environment_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROBE_OUT"
ssh -o ConnectTimeout=10 xdf-45 'bash -s' \
  < scripts/probe_experiment_host.sh 2>&1 | tee "$PROBE_OUT/host45.txt"
ssh -o ConnectTimeout=10 xdf-35 'bash -s' \
  < scripts/probe_experiment_host.sh 2>&1 | tee "$PROBE_OUT/host35.txt"
printf 'PROBE_OUT=%s\n' "$PROBE_OUT"
```

也可把该脚本复制到已登录的服务器终端执行。脚本打印 Python/依赖/可编辑源码位置、驱动、磁盘、模型元数据位置，不安装依赖、不启动模型。

## 2. 根据盘点决定迁移方式

**先确认共享存储。** 两台机器都出现 `/data` 或同一路径不证明共享。可在用户自己的迁移目录创建唯一标记文件，再在另一台核对内容；只删除自己创建的标记。确认为共享后，同路径环境/权重可直接复用，但仍要验证 35 的动态库、驱动与 GPU kernel。

**如果不是共享存储，迁移分四部分：**

1. Conda 环境；2. 本轮项目代码和 MuSiQue 数据；3. 完整、已物化的 SFT HF 模型；4. 环境依赖的本地源码/外部 CUDA 库。

源环境复制到 35 的同一绝对路径可减少硬编码前缀问题，但仍有可编辑源码、动态库、系统 ABI 和用户目录依赖。不要只 `scp site-packages` 或覆盖目标已有环境。

若源环境已具备 `conda-pack`，优先打包后在最终路径解包，再运行 `conda-unpack`；系统平台须匹配。[官方迁移说明](https://conda.github.io/conda-pack/)

打包前记录 `pip freeze` 和可编辑安装位置。存在 editable 包时先审核路径，必要时使用本机 `conda-pack --help` 确认 `--ignore-editable-packages` 支持，再配套同步源码；该选项不会把外部源码自动打进归档。不能为了打包成功随意忽略缺文件错误。尚未核验实际环境，当前不提供会直接覆盖目标的迁移命令。

## 3. 模型和数据要单独核验

迁移工具兼容性记录（2026-09-30）：45使用`pip --target pack_tools conda-pack==0.8.1`时，解析到`setuptools==84.0.0`，工具导入失败：`ModuleNotFoundError: pkg_resources`。Setuptools从82起移除此模块；修复仅针对独立迁移工具目录，固定`conda-pack==0.8.1`和`setuptools==80.9.0`，先验证导入和`--version`，不降级原训练环境。新建兼容工具目录可避免旧target目录残留不同版本metadata。

本机独立Python3.12 venv验证：上述固定组合可导入`conda_pack`及`pkg_resources`；`python -m conda_pack.cli --version`输出`conda-pack 0.8.1`。此前给出的`python -m conda_pack`写法不正确，该版本没有`__main__`；应使用`conda-pack`入口或`python -m conda_pack.cli`。这只验证工具启动，未验证45实际环境打包。

- 首选已有 SFT-874 的完整 HF 导出；原始 Qwen3.5-9B 作为可选 smoke 对照。
- `hf_partial`、Megatron `iter_*` 目录、只含权重的 `weight_v*` 不能默认当成可直接部署的完整模型。
- 校验 config、tokenizer/chat template、权重索引及索引引用的所有 shards。绝对软链指向 45 的 base 模型时，需物化并记录来源，再传输。
- 不复制全部历史训练输出；仅选定模型与本轮需要的数据/源码。
- 35 数据先比对 `stats.json` 和 task/corpus hash，保持 train/discovery/confirmation 分组规则。
- 用户要求本机实验代码提交推送后由35克隆/拉取`slime-rewrite`；数据仍从本机rsync。实验记录保存实际Git commit，不能只记录分支名。35路径为`/local_data/zhangyonglin/ResearchAgent-RL`，获取命令见`effect_space_server_runbook_cn.md`；尚未实现的新诊断驱动不能因拉取而被视为已实现。

## 4. 35 上的验证顺序

1. 检查空闲 GPU、端口、可用磁盘、glibc 与驱动。
2. 验证 Python、torch、vLLM 和检索依赖；对照45记录，查找外部路径。
3. 在选定 GPU 上做 bf16 GPU tensor 操作及同步，验证驱动实际可运行 CUDA kernel。环境打包不包含宿主驱动。
4. 用一张空闲卡启动固定 SFT 权重，单独端口，记录完整启动参数。核验 `/v1/models` 后发送一条实际 action 请求。
5. CPU tests / 数据与检索 smoke。
6. 新 runtime-efficiency 驱动实现并测试完成后，执行实验计划 P1；按真实吞吐和错误率决定 P2。

首轮冻结推理用 1 张 A800 即可开始；8 张卡可供后续独立模型或分片并发，不需要为诊断自动启动八卡训练。结果、cache 和临时文件放 `/data/zhangyonglin/research-agent-rl-data/runtime_efficiency`；仅在项目 `results` 下创建独立子链接。

## 完成清单

- [x] 用户SSH登录成功、两端盘点已提供
- [x] 挂载输出已确认当前35未共享45的`/data`
- [x] 推理环境迁移完成，归档约7.02GB且SHA256校验通过；35目标`/local_data/zhangyonglin/conda_envs/research-agent-runtime`
- [x] 35环境导入与bf16矩阵乘通过：torch2.11.0+cu129、CUDA12.9、vLLM0.23.0+cu129、A80080GB；用户提供实际`GPU kernel: OK`
- [ ] 若使用Vime训练，外部editable源码和CUDA开发库需另外同步、核验
- [ ] SFT 模型完整，数据与代码 hash 已保存
- [ ] 35上的模型服务和实际action请求通过
- [ ] 新诊断实现/CPU tests 通过
- [ ] P1 实验样本、错误分类及成本记录完成

用户已完成环境迁移与35 GPU kernel验证。SFT模型源选择此前3K评测使用的完整导出：`/data/zhangyonglin/research-agent-rl-data/outputs/hotpotqa_eval_sft874_full_seed20260713_20260722_154042/iter_0000874/hf_merged`。模型、最新本机代码、MuSiQue数据尚待同步；正式新诊断驱动尚待实现。用户执行服务器命令，未将本机测试称作远端实验。

## 本机代码发布验证（2026-09-30）

- 使用独立Python3.12测试环境安装`requirements.txt`，不修改服务器环境。
- `test_effect_space.py`、`test_effect_space_continuation_pilot.py`、`test_decision_value_pilot.py`、`test_search_tool.py`：23项通过。
- 三个诊断Python脚本的`--help`、盘点shell脚本`bash -n`均通过。
- 全量测试：81 passed、2 skipped、1 failed。失败项为`TestHotpotQAPreparation.test_build_split_creates_isolated_train_and_eval_corpora`，检索`evidence`返回空列表；在未改动的`3aecc30`代码快照、相同依赖环境下复现，属于已有问题，本次未改动该测试或CorpusStore。不能将本次验证表述为全量测试通过。
- 本轮发布已有冻结策略诊断代码和规划/运行文档；`runtime_efficiency_pilot.py`及任意前缀KEEP/REPLAN/ALT的完整新实验驱动尚未实现。
