# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 16-bit RGGB Bayer raw（`train_data/Data/<scene>/<iso>/sensorN_raw.png`）做
**Neighbor2Neighbor** 自监督单帧去噪。残差预测、FP16，跑在 RTX 2070 上。
PyQt5 GUI 实时显示 loss 曲线（log-log）、64×64 patch 的 noisy / denoised
对比、训练进度，结束后一键把全量推理结果写入 `result/`。

代码库里只保留两条干净的训练路径：

| 路径 | 损失 | 何时用 | 推理代价 |
|---|---|---|---|
| **生产 baseline** (`run_train.py`) | `L1 + 0.03·TV` (`n2n.losses.l1_plus_tv`) | 默认部署 / 想最快出 ckpt | 1× |
| **★ 推荐感知方案** (`train_vgg.py`) | robust 训练 (EMA + R2R + mask + jitter) + `vgg_l1_eatv` (`n2n.losses_perceptual.vgg_l1_eatv`) | 想让残留噪声"看起来像自然噪声"，符合人眼 | 1×（VGG 只在训练时） |

> **整个项目的实验过程、4 个阶段的所有尝试（初始扫描 → TV 调优 →
> 7h 92 候选 → 3h 鲁棒 → 6 候选感知损失）和每个阶段的结论 / 经验，
> 全在 [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)**。
> 历史 HTML 报告、中间过程图、被淘汰的方案代码都在本地
> `_archive_local/`（已 gitignored），随时可按 journey 文档索引找回。

## 0. 快速上手

新机器部署见 [`QUICKSTART.md`](QUICKSTART.md)。一行话：

```bash
pip install -r requirements.txt
python benchmark.py             # 跑 3 min 看本机 sps，估算合适训练时长
python run_train.py             # 启 GUI 真正训练（默认 L1+0.03·TV）
```

想跑 ★ VGG 推荐方案：

```bash
python train_vgg.py
```

## 1. 数据规格

- raw：`uint16`, 1280×1088，单通道 Bayer，**RGGB** 模式
- isp：`uint8` BGR，与 raw 同尺寸，仅作视觉参考，**不参与训练**
- 7 个场景 × 10 个 ISO × 2 个 sensor = 140 对
- 网络输入：`raw / 65535` → `float`（FP16），4 通道打包形状 `(B, 4, H/2, W/2)`，
  通道顺序 `R, Gr, Gb, B`
- ISO=16 是最噪的（自监督训练通常在最高 ISO 上效果最好）

## 2. 显示流水线（避免 staircase 的关键）

raw → 显示，全程 16-bit / float32，最后才量化到 8-bit（实现在
`n2n/raw_utils.py::raw_to_display`）：

```
raw_uint16
  -> 16-bit black-level subtract            (raw_to_linear_bgr)
  -> 16-bit demosaic (cv2 BayerRG2BGR)
  -> float32 BGR 0-255
  -> gray-world WB                          (raw_to_display)
  -> gamma 2.2
  -> uint8                                  (only here!)
```

之前的实现在 demosaic 之前就 `astype(uint8)` 了，导致大平面出现明显色阶。
完整修复路径见 `docs/EXPERIMENTS_JOURNEY.md` 阶段 1。

## 3. 训练

### A. 生产 baseline (L1 + 0.03·TV)

```bash
python run_train.py                   # 一键 GUI（推荐）
# 或 headless：
python -c "from n2n.trainer import Trainer, TrainConfig; Trainer(TrainConfig()).run()"
```

GUI 顶部可调：`data root` / `train`（训练时长，秒）/ `batch` / `patch` / `lr` / `FP16` 开关。

训练中：

- loss 曲线（log-log）实时更新
- 64×64 patch 通过显示流水线转 RGB → nearest-neighbour 放大显示左右 noisy / denoised 对比
- 进度条 + 状态栏
- 训练结束后点 `Generate result/ comparison`：对全量 raw 推理，写入
  `result/<scene>/<iso>/sensorN_compare.png`（noisy / denoised / vendor ISP 三联）
  + `sensorN_denoised_raw.png`（16-bit 去噪 raw）

`n2n/trainer.py` 的 `TrainConfig` 锁定为：

```python
loss_name    = "l1_plus_tv"
loss_kwargs  = {"lam": 0.03}         # 平面/角落清晰 + 边缘保留 的甜蜜点
patch_size   = 256                   # packed-pixel，对应 512 Bayer
batch_size   = 6
unet_depth   = 3
base_channels= 48                    # ~3 M 参数, ~12 MB ckpt
use_fp16     = True
n2n_lambda   = 0.0                   # 不加 N2N 一致性正则，纯 L1+TV 主项
train_seconds= 1200                  # 20 min（推荐）
intermediate_save_seconds = (120, 300, 600)   # 自动存 2/5/10 min 快照
```

### B. ★ VGG 推荐方案

```bash
python train_vgg.py                  # 训两个候选
python train_vgg.py --only P4_vgg    # 只训 VGG
python train_vgg.py --ckpt-dir my_ckpts/   # 自定义输出目录
```

会按顺序训：

```
P0_baseline   robust + L1+0.03·TV       ~5 min  (公平对照)
P4_vgg        robust + VGG feature L1   ~7 min  ★
```

输出（默认）：
- `ckpts/P0_baseline.pt`
- `ckpts/P4_vgg.pt`
- `ckpts/logs/<name>.log`

两个 ckpt 推理路径与 baseline 完全一样（用 `n2n.infer.run_inference_set`），
**BPU / ONNX / TRT 部署 0 改动**——VGG 只在**训练**时走一次 frozen forward。

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素，
  整批共享同一 mask。子图分辨率减半，通道维不变。
- **L1 + 0.03·TV 损失**（`n2n/losses.py`）：
  ```
  main = L1(g1 - f(g1), g2) + 0.03 * TV(g1 - f(g1))
  ```
  TV 项压平大平面（暗角彩噪），L1 主项保留细节。
- **VGG 感知损失**（`n2n/losses_perceptual.py::vgg_l1_eatv`）：
  ```
  main = 4·L1(Y_d, Y_t) + L1(Co/Cg) + 0.03·EATV(d) + 0.05·L1(VGG(d), VGG(t))
  ```
  YCoCg 4:1 加权对齐 HVS 的亮度/色度敏感度，EATV 是边缘自适应 TV，VGG-16
  特征 L1 用 relu1_2 / relu2_2 / relu3_3 三层（=LPIPS 同款）。
- **Robust 训练机制**（`n2n/robust_trainer.py`，VGG 走这里）：
  EMA 自蒸馏 + R2R 噪声注入 + 随机像素掩码 + 亚像素抖动，全部**训练时**
  生效，推理路径不变。诊断和动机见 `docs/EXPERIMENTS_JOURNEY.md` 阶段 5。
- **残差头零初始化**：`UNet.out_conv` 权重 / 偏置全 0，网络起始等价于恒等。
- **FP16**：`torch.amp.autocast` + 自动 dtype 选择（Ampere+ 走 bf16，Turing 走 fp16）。
- **GUI 与训练解耦**：训练跑在 `QThread`，通过 Qt signal 把 `StepInfo` /
  `PreviewInfo` 发回 UI 主线程。

## 5. 工程结构

```
new_denoise/
├── n2n/                              # 生产代码（生效）
│   ├── raw_utils.py                  # 16-bit display pipeline + I/O + Bayer pack
│   ├── n2n_sampler.py                # vectorised pair sampling
│   ├── dataset.py                    # CPU DataLoader fallback
│   ├── gpu_dataset.py                # GPU-resident 数据集（默认）
│   ├── model.py                      # UNet (residual head)
│   ├── losses.py                     # L1+0.03·TV + 7 备选 + perceptual lazy 注册
│   ├── losses_perceptual.py          # ★ VGG 特征损失 + luma/chroma + EATV
│   ├── trainer.py                    # 主 trainer (run_train.py 走这里)
│   ├── robust_trainer.py             # robust 训练 (EMA + R2R + mask + jitter; train_vgg.py 走这里)
│   └── infer.py                      # tile 推理 + ImageJ-stack 输出
├── run_train.py                      # 一键启 GUI（生产 baseline，L1+0.03·TV）
├── train_vgg.py                      # ★ 一键训 P0_baseline + P4_vgg 两个 ckpt
├── train_gui.py                      # PyQt5 GUI 实现
├── benchmark.py                      # 3 分钟吞吐量基准
├── docs/
│   └── EXPERIMENTS_JOURNEY.md        # ★ 整个项目的来龙去脉 + 经验
├── requirements.txt
├── README.md                         # （本文件）
├── QUICKSTART.md                     # 部署到新机器的步骤
│
│   # ↓ 以下都不被 git 跟踪，本地保留（已 gitignored）
├── train_data/                       # 训练数据（自备）
├── checkpoints/                      # run_train.py 输出（自动建）
├── ckpts/                            # train_vgg.py 输出（自动建）
├── result/                           # 推理对比图（自动建）
└── _archive_local/                   # 历史 HTML 报告 / 中间过程 PNG / 被淘汰的脚本 / 大件归档
```

## 6. 资源占用参考（RTX 2070 + FP16）

| 配置 | step/sec | 显存 | 备注 |
|---|---|---|---|
| L1+0.03·TV (depth=3, patch=256, batch=6) | ~55–60 | ~5.5 GB | 1 epoch ≈ 3.0 s |
| robust + VGG feature L1                  | ~42–48 | ~6.5 GB | VGG forward ~25% slower |

整图 1280×1088 推理 + 512 tile + 32 overlap，单帧约 0.3 s，与 baseline 一致。
