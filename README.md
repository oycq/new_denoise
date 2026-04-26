# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 16-bit RGGB Bayer raw（`train_data/Data/<scene>/<iso>/sensorN_raw.png`）做
**Neighbor2Neighbor** 自监督单帧去噪。残差预测、FP16，跑在 RTX 2070 上。
PyQt5 GUI 实时显示 loss 曲线（log-log）、64×64 patch 的 noisy / denoised
对比、训练进度，结束后一键把全量推理结果写入 `result/`。

## 项目当前的损失 = `L1 + 0.05·VGG`

> 经过 4 个阶段的实验（初始扫描 → TV 调优 → 7h 92 候选 → 3h 鲁棒 →
> 6 候选感知损失，全程见 [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)），
> 又用 `tmp/` vs `tmp2/` 做了一次 "kitchen-sink (EMA + R2R + mask + jitter +
> luma/chroma + EATV + VGG)" 与 "minimal (纯 L1+VGG)" 的并排对比，
> 视觉上看不出区别——多余的机制只让每步变慢、不带来肉眼可见的提升。
>
> 所以现在代码库只保留这一条路径：
>
> ```
> loss = L1(den, target) + lam_vgg · vgg_perceptual_loss(den, target)
> lam_vgg = 0.05
> ```
>
> 历史 HTML 报告、过程图、被淘汰的脚本都在本地
> `_archive_local/`（已 gitignored）；研究产物 `tmp/` `tmp2/` `tmp3/`
> 也保留在仓库根目录作为对比记录，不参与生产路径。

## 0. 快速上手

新机器部署见 [`QUICKSTART.md`](QUICKSTART.md)。一行话：

```bash
pip install -r requirements.txt
python benchmark.py             # 跑 3 min 看本机 sps，估算合适训练时长
python run_train.py 1200        # 启 GUI 训 20 min（默认 L1 + 0.05·VGG）
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

GUI 版（推荐）：

```bash
python run_train.py             # 默认 5 分钟
python run_train.py 1200        # 20 分钟
```

GUI 顶部能调 `data root` / `train`（秒）/ `batch` / `patch` / `lr` / `FP16` 开关。

训练中：

- loss 曲线（log-log）实时显示 L1 / VGG / total 三条线
- 64×64 patch 通过显示流水线转 RGB → nearest-neighbour 放大显示左右 noisy / denoised 对比
- 进度条 + 状态栏
- 训练结束后点 `Generate result/ comparison`：对全量 raw 推理，写入
  `result/<scene>/<iso>/sensorN_compare.png`（noisy / denoised / vendor ISP 三联）
  + `sensorN_denoised_raw.png`（16-bit 去噪 raw）

Headless 版：

```bash
python -c "from n2n.trainer import Trainer, TrainConfig; Trainer(TrainConfig()).run()"
```

`n2n/trainer.py` 的 `TrainConfig`（默认值）：

```python
lam_vgg      = 0.05                  # L1 + 0.05·VGG
patch_size   = 256                   # packed-pixel，对应 512 Bayer
batch_size   = 6
unet_depth   = 3
base_channels= 48                    # ~3 M 参数, ~12 MB ckpt
use_fp16     = True
train_seconds= 1200                  # 20 min（推荐）
intermediate_save_seconds = (120, 300, 600)   # 自动存 2/5/10 min 快照
```

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素，
  整批共享同一 mask。子图分辨率减半，通道维不变。
- **L1 + 0.05·VGG 损失**（`n2n/trainer.py` 内联 + `n2n/losses_perceptual.py::vgg_perceptual_loss`）：
  ```
  loss = L1(g1 - f(g1), g2) + 0.05 · L1(VGG_φ(d), VGG_φ(t))
  ```
  VGG-16 的 `relu1_2 / relu2_2 / relu3_3` 三层（=LPIPS 同款），frozen，
  fp32 内部计算（autocast-safe）。**只在训练时**走一次 VGG forward，
  推理路径不变 → BPU / ONNX / TRT 部署 0 改动。
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
│   ├── losses_perceptual.py          # vgg_perceptual_loss（唯一非平凡 loss）
│   ├── trainer.py                    # 训练循环：L1 + lam_vgg·VGG
│   └── infer.py                      # tile 推理 + ImageJ-stack 输出
├── run_train.py                      # 一键启 GUI
├── train_gui.py                      # PyQt5 GUI 实现
├── benchmark.py                      # 3 分钟吞吐量基准
├── docs/
│   └── EXPERIMENTS_JOURNEY.md        # 整个项目的来龙去脉 + 经验
├── requirements.txt
├── README.md                         # （本文件）
├── QUICKSTART.md                     # 部署到新机器的步骤
│
│   # ↓ 以下是研究产物 / 不被 git 跟踪的本地大件
├── tmp/                              # 20-min kitchen-sink 训练历史快照（保留作对比）
├── tmp2/                             # 10-min minimal L1+VGG 训练（与 tmp/ 直接对比）
├── tmp3/                             # 两条路径的并排 PNG 输出
├── train_data/                       # 训练数据（自备，git 忽略）
├── checkpoints/                      # run_train.py 输出（自动建，git 忽略）
├── result/                           # 推理对比图（自动建，git 忽略）
└── _archive_local/                   # 历史 HTML 报告 / 中间过程 PNG / 被淘汰的脚本（git 忽略）
```

## 6. 资源占用参考（RTX 2070 + FP16）

| 配置 | step/sec | 显存 | 备注 |
|---|---|---|---|
| L1 + 0.05·VGG (depth=3, patch=256, batch=6) | ~42–48 | ~6.5 GB | VGG forward ~25% 慢于纯 L1 |

整图 1280×1088 推理 + 512 tile + 32 overlap，单帧约 0.3 s，VGG 不参与推理。
