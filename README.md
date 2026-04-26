# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 16-bit RGGB Bayer raw（`train_data/Data/<scene>/<iso>/sensorN_raw.png`）做
**Neighbor2Neighbor** 自监督单帧去噪。FP16，跑在 RTX 2070 / 4090 上。
PyQt5 GUI 实时显示 loss 曲线（log-log）、64×64 patch 的 noisy / denoised
对比、训练进度，结束后一键把全量推理结果写入 `result/`。

## 当前损失 = N2N 论文原配方（CVPR 2021）

```
L = ‖f(g₁) − g₂‖²  +  γ(t) · ‖(f(g₁) − g₂) − (g₁(f(y)) − g₂(f(y)))‖²
```

- `f` 是网络（**非残差** UNet，输出即去噪结果）
- `g₁, g₂` 是 N2N 子采样
- `y` 是完整 noisy 输入
- `γ(t)` 从 0 线性升到 2 ——前者是重建项（让 f(g₁) 逼近 g₂ 的均值 = clean），
  后者是**一致性正则**（让子采样推理和全图推理输出一致，是 N2N 区别于
  其他自监督方法的核心创新）

> 这是经过 5 个阶段实验后留下的**唯一**配方，详细历程见
> [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)。
> 简短版：之前的 L1 + VGG perceptual / 各种 mode penalty / Gr-Gb 一致性
> 都是给 2×2 grid artifact 打补丁，最终发现**根因是缺少 N2N 论文的
> 一致性正则项**，加回来 + 用非残差网络（N2V2 anti-checkerboard fix）+
> BlurPool 下采样 + PixelShuffle 上采样后，artifact 自然消失，无需
> 任何额外 loss term。

## 0. 快速上手

新机器部署见 [`QUICKSTART.md`](QUICKSTART.md)。一行话：

```bash
pip install -r requirements.txt
python benchmark.py             # 跑 3 min 看本机 sps
python run_train.py 600         # 启 GUI 训 10 min（最低稳定值，推荐起点）
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

## 3. 训练

GUI 版（推荐）：

```bash
python run_train.py             # 默认 10 分钟
python run_train.py 1200        # 20 分钟
```

GUI 顶部能调 `data root` / `train`（秒）/ `batch` / `patch` / `lr` / `FP16` 开关。

训练中：

- loss 曲线（log-log）实时显示 **rec L2 / reg L2 / total** 三条线
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
patch_size   = 256                   # packed-pixel，对应 512 Bayer
batch_size   = 6
unet_depth   = 3
base_channels= 48                    # ~3 M 参数，~12 MB ckpt
lr           = 3e-4                  # 论文默认
use_fp16     = True
train_seconds= 600                   # 10 min（最低稳定值）
gamma_final  = 2.0                   # γ 在训练结束时的目标值
intermediate_save_seconds = (300, 600)
```

> ⚠️ **训练时长 ≥ 600 s**。非残差 UNet 没有零初始化的恒等先验，
> 5 min 以下有概率出现"灾难棋盘"（output 是大片彩色 2×2 网格）；
> 10 min 是稳定区，15-20 min 给 ~5-10% 渐进改善。

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素。
  子图分辨率减半，通道维不变。
- **N2N 论文 loss**（`n2n/trainer.py` 内联）：
  ```
  diff      = f(g₁) − g₂
  exp_diff  = g₁(f(y)) − g₂(f(y))     [no_grad]
  loss      = mean(diff²) + γ(t) · mean((diff − exp_diff)²)
  ```
  第二项强制子采样推理和全图推理输出一致——没这一项的训练等价于普通自监督，
  会出可见的 2×2 grid artifact（参见 `docs/EXPERIMENTS_JOURNEY.md` 阶段 7）。
- **非残差 UNet** (`n2n/model.py`)：`denoise(x) = forward(x)`，不是
  `x − forward(x)`。N2V2 (Hoeck et al. 2022) 论文证明残差头是
  blind-spot/N2N 自监督方法 checkerboard 的来源。
- **BlurPool 下采样**：3×3 [1,2,1] 可分离模糊 + avg-pool，替代 max-pool 防混叠。
- **PixelShuffle 上采样**：1×1 conv → channel-to-space rearrange，
  替代 stride-2 ConvTranspose 防 checkerboard。
- **FP16**：`torch.amp.autocast` + 自动 dtype 选择（Ampere+ 走 bf16，Turing 走 fp16）。
- **GUI 与训练解耦**：训练跑在 `QThread`，通过 Qt signal 把 `StepInfo` /
  `PreviewInfo` 发回 UI 主线程。

## 5. 工程结构

```
new_denoise/
├── n2n/                              # 生产代码
│   ├── raw_utils.py                  # 16-bit display pipeline + I/O + Bayer pack
│   ├── n2n_sampler.py                # vectorised pair sampling
│   ├── dataset.py                    # CPU DataLoader fallback
│   ├── gpu_dataset.py                # GPU-resident 数据集（默认）
│   ├── model.py                      # 非残差 UNet (BlurPool + PixelShuffle)
│   ├── trainer.py                    # 训练循环：L2 + γ·L_reg
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
│   # ↓ 以下不被 git 跟踪
├── train_data/                       # 训练数据（自备）
├── checkpoints/                      # run_train.py 输出
├── result/                           # 推理对比图
└── _archive_local/                   # 历史报告 / 实验产物
```

## 6. 资源占用参考（RTX 4090 + FP16）

| 配置 | step/sec | 显存 | 备注 |
|---|---|---|---|
| 非残差 UNet + L2 + γ·L_reg (depth=3, patch=256, batch=6) | ~110 | ~2 GB | 第二个 forward (no_grad) 让步进慢一倍但损失项是论文原版 |

整图 1280×1088 推理 + 512 tile + 32 overlap，单帧约 0.3 s。
