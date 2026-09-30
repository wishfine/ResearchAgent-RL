# 假期前优先启动MuSiQue GRPO：最短路径

日期：2026-09-30。用户要求优先RL，不等待embedding和192条协议对照。

## 已有与缺失

35已有完整HF模型`/local_data/zhangyonglin/models/ResearchAgent-Qwen3.5-9B-SFT874`（4shards，17.98GiB权重）、推理环境、MuSiQue数据和CUDA12.9工具链。**不重复传HF模型、不换新框架、不先重跑SFT。**

用户确认35没有`/local_data/zhangyonglin/vime-src`。需要从45同步已打补丁的Vime/Megatron源码，以及Megatron格式的SFT874 checkpoint。HF导出可推理，但旧Vime入口的`--load`/`--ref-load`不能直接替换成HF目录。

35外部flash-attn因为glibc不匹配已经卸载。新入口用`attention-backend=auto`让TE选择可用路径，**不是证明cuDNN训练一定可用**；TE/FusedAttention/GDN/backward仍需真2更新验收。若torch/TE/mbridge/TMS导入报ABI错，先修训练算子，不盲目启动200步。

## 第一轮锁定设置

- SFT874 Actor + 同初始化冻结Reference，重新创建GRPO optimizer/RNG。
- GPU2–5 Actor，TP2/DP2；GPU6–7 rollout，TP2，6卡全参。GPU0/1留着；忙卡会拒绝运行，不自动杀别的进程。
- MuSiQue train，split-wide SQLite BM25；**没有embedding依赖**。官方dev不进入训练。
- READ全文，adaptive多跳prompt含CITE前置条件；最大15动作，每次生成cap512，temperature0.7。
- 每rollout group 1题×8samples；global batch8、micro batch1。200rollout意味着目标200个group（名义1600条轨迹），异步buffer/filter实际消耗以日志为准，不称1600个不同问题。
- KL0.02、现有reward公式、format0.20、invalid0.50、no-answer0.20、step0.005。降低step惩罚是为了不继续强化旧四步停止习惯，属于预注册配置选择，不是已验证最优值。
- lr1e-6、BF16 moments、关闭FP32梯度累加，继承原已验证6卡训练设置；disk weight sync兼容vLLM0.23，每25group保存。
- 保留旧训练入口默认行为；新参数只在新MuSiQue入口opt-in。

这是新环境上的**GRPO baseline**，不是已实现/已验证的新算法；不把检索或prompt变化的收益算成RL算法创新。后续算法A/B保持相同数据、prompt、工具和reward。

## 1. 35同步45训练源码和SFT训练checkpoint

在35当前终端运行（需要45登录密码；可续传，不用--delete）。checkpoint可能比HF模型大，因为distcp分片可能混有optimizer状态；实际大小先打印，不假定18GB：

```bash
(
set -euo pipefail
PROJECT="/local_data/$USER/ResearchAgent-RL"
BASE="/local_data/$USER/research-agent-rl-data"
cd "$PROJECT"
git pull --ff-only origin slime-rewrite
mkdir -p "/local_data/$USER/vime-src" "$BASE/checkpoints/sft874"

rsync -avh --partial --progress \
  --exclude='.git' --exclude='__pycache__' \
  zhangyonglin@172.22.0.45:/data/zhangyonglin/vime-src/vime \
  zhangyonglin@172.22.0.45:/data/zhangyonglin/vime-src/Megatron-LM \
  "/local_data/$USER/vime-src/"

ssh zhangyonglin@172.22.0.45 \
  'du -sh /data/zhangyonglin/research-agent-rl-data/outputs/vime_hotpotqa_sft7k_20260721_162728/checkpoints/iter_0000874'

rsync -avh --partial --progress \
  --include='/latest_checkpointed_iteration.txt' \
  --include='/iter_0000874/' --include='/iter_0000874/***' --exclude='*' \
  zhangyonglin@172.22.0.45:/data/zhangyonglin/research-agent-rl-data/outputs/vime_hotpotqa_sft7k_20260721_162728/checkpoints/ \
  "$BASE/checkpoints/sft874/"

bash scripts/probe_vime_training_35.sh
)
```

复制保留patched source，不重新checkout远端分支导致patch丢失；不安装/覆盖另一个项目环境。`TRAIN_IMPORTS_READY`只证明路径/导入，不证明GPU训练就绪。如果不通过，发完整probe输出，尤其TE/TMS的`.so`/GLIBC错误。

## 2. 真2更新验收，马上看gradient和权重同步

先确认GPU2–7空闲。不调用global `ray stop`；MuSiQue入口默认`RAY_AUTO_STOP=0`，避免误停其他Ray集群。训练结束可能仍有Ray进程/actor占卡，需要明确清理本run后再开始下一run；不要直接杀所有python。

```bash
(
set -euo pipefail
cd /local_data/$USER/ResearchAgent-RL
BASE="/local_data/$USER/research-agent-rl-data"
RUN_DIR="$BASE/outputs/vime_musique_grpo_smoke2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
GPU_IDS=2,3,4,5,6,7 NUM_ROLLOUT=2 SAVE_INTERVAL=2 RUN_DIR="$RUN_DIR" \
  nohup bash scripts/run_vime_musique_grpo.sh >"$RUN_DIR/driver.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid.txt"
printf '%s\n' "$RUN_DIR" > "$BASE/latest_musique_grpo.txt"
echo "RUN_DIR=$RUN_DIR"
)
```

重新开终端也能看：

```bash
BASE="/local_data/$USER/research-agent-rl-data"
RUN_DIR="$(cat "$BASE/latest_musique_grpo.txt")"
echo "$RUN_DIR"
tail -n 100 "$RUN_DIR/driver.log"
grep -E 'train/|raw_reward|Updating rollout weights|Reloading and processing|Job .*succeeded|Traceback|OutOfMemory|failed' \
  "$RUN_DIR/driver.log" | tail -n 100
```

验收需要：真实train step/grad及非NaN reward/loss、至少一次训练后的disk reload、job succeeded。只有preflight imports或vLLM载模不够。全组advantages=0需要看8条原始reward是否相同，不能强行宣布有学习信号。

## 3. 200group基线

smoke2通过且本run的Ray/卡占用已清理后，以相同配置另起新RUN_DIR，将`NUM_ROLLOUT=2 SAVE_INTERVAL=2`换成`NUM_ROLLOUT=200 SAVE_INTERVAL=25`。新run会**重新从SFT874开始**，不是继续smoke optimizer。六卡拓扑不变；不要回到之前OOM的四卡全参配置。

本轮优先取得真实训练数据；embedding对照、全文协议的192episode和新算法消融都移出今天的启动关键路径，但不能在最终论文里省掉对照。先记录前5–10个group的真实耗时再估200group总时间，不能拿推理吞吐或HotpotQA旧耗时直接保证MuSiQue时长。

本机验证：全量109 passed、2 skipped；入口新增3项行为测试独立审查复核通过，三脚本shell语法通过。已有HF路径保持复用；启动前prompt内容逐条与当前train tasks核对，不覆盖不匹配旧文件。Vime的loss-mask/log-prob拼接算法未修改，GPU训练及optimizer/权重同步仍待35真2更新验收。

## 同步后的实际阻塞与定向修复

用户实际日志确认：源码已同步43.35MB；SFT训练checkpoint含8个distcp分片、common.pt、.metadata与metadata.json，约144.77GB已同步（非HF模型大小）。路径都通过，torch/vLLM/Ray/FLA/TMS和项目rollout均导入。

失败的TE/mbridge/megatron.core都指向同一`transformer_engine_torch*.so`需要GLIBC_2.32；35宿主不满足，不能使用45编译的这个binding。系统libc不升级，保留TE2.10 Python与cu12核心，仅在35用原2.10源码重建torch binding。[TE2.10官方FORCE_BUILD入口](https://github.com/NVIDIA/TransformerEngine/blob/v2.10/transformer_engine/pytorch/setup.py)支持`NVTE_PYTORCH_FORCE_BUILD=TRUE`，避免再次下载预编译wheel。

CUDA并未缺失：用户之前实测nvcc12.9/FlashInfer成功时的CUDA_HOME是`/local_data/zhangyonglin/cuda-toolkit-12.9`。本轮probe/入口错误套用了45嵌套目录，已修正两处默认值并加回归测试。显式CUDA_HOME仍可覆盖。

在35只传编译源包，不再传模型/checkpoint：

```bash
(
set -euo pipefail
cd /local_data/$USER/ResearchAgent-RL
git pull --ff-only origin slime-rewrite
mkdir -p /local_data/$USER/vime-src/wheels
rsync -avh --partial --progress \
  zhangyonglin@172.22.0.45:/data/zhangyonglin/vime-src/wheels/transformer_engine_torch-2.10.0.tar.gz \
  /local_data/$USER/vime-src/wheels/

BASE="/local_data/$USER/research-agent-rl-data"
RUN_DIR="$BASE/runtime_efficiency/te_torch_native35_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
RUN_DIR="$RUN_DIR" nohup bash scripts/repair_te_torch_35.sh \
  >"$RUN_DIR/build.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/pid.txt"
printf '%s\n' "$RUN_DIR" > "$BASE/runtime_efficiency/latest_te_native35.txt"
echo "BUILD_DIR=$RUN_DIR"
)
```

查看：

```bash
BASE="/local_data/$USER/research-agent-rl-data"
BUILD_DIR="$(cat "$BASE/runtime_efficiency/latest_te_native35.txt")"
tail -n 60 "$BUILD_DIR/build.log"
```

脚本离线构建，固定原source SHA、三包2.10版本、Torch CUDA12.9、系统g++，只更新torch binding，不更新torch/vLLM。先构建wheel、readelf验证所需GLIBC不高于宿主，再备份旧.so、安装并测试导入。失败保留build.log，旧binding在构建失败时不动；安装后若导入失败不自动删除/降级其他包。已移除的外部FlashAttention不在本次修复范围。

编译不使用GPU；之后2步GRPO才验证TE/GDN/backward/optimizer。用户最新盘点GPU0=56140MiB、GPU1=632MiB、GPU2=8298MiB、GPU3–7=13MiB：GPU2在忙，启动六卡前先确认PID/用途，不依据显存猜测并自动杀进程。
