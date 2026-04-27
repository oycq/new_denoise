# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 16-bit RGGB Bayer raw（`train_data/Data/<scene>/<iso>/sensorN_raw.png`）做
**Neighbor2Neighbor** 自监督单帧去噪。FP16，跑在 RTX 2070 / 4090 上。
PyQt5 GUI 实时显示 loss 曲线（log-log）、64×64 patch 的 noisy / denoised
对比、训练进度，结束后一键把全量推理结果写入 `result/`。

## 当前损失 = L1 + 0.01 · TV

```
L = L1(f(g₁), g₂)  +  0.01 · TV(f(g₁))
```

- `f` 是网络（残差 UNet，`denoise(x) = x − forward(x)`）
- `g₁, g₂` 是 N2N 子采样
- `TV(·)` 是去噪输出上的 anisotropic 总变差：`mean(|∂x den|) + mean(|∂y den|)`

> 这是经过 5 个阶段实验 + 1 次 λ 扫描 + 1 次训练时长扫描后留下的**唯一**配方。
> 完整历程见 [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)。
> 之前主干里出现过的 paper N2N (L2 + γ·L_reg)、L1 + 0.05·VGG、multi-scale
> Charbonnier、robust_trainer (EMA + R2R) 等方法的 recipe / 复活步骤都
> 沉淀到 [`docs/archived_recipes.txt`](docs/archived_recipes.txt)。

## 0. 快速上手

新机器部署见 [`QUICKSTART.md`](QUICKSTART.md)。一行话：

```bash
pip install -r requirements.txt
python benchmark.py             # 跑 1 min 看本机 sps
python run_train.py 600         # 启 GUI 训 10 min
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

- loss 曲线（log-log）实时显示 **L1 + λ·TV** 单条线
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
lr           = 2e-4
use_fp16     = True
train_seconds= 600                   # 10 min（loss 已平台，视觉略好于 5 min）
tv_lambda    = 0.01                  # TV 系数；0.05 太狠、0.03 仍有平整化、0.01 视觉甜点
intermediate_save_seconds = (300, 600)
```

> **关于训练时长**：5 min 时 loss 已经落到 ~0.0162，10 min 几乎再无下降；
> 但视觉上 10 min 在 ISO=16 暗区 / 角落 lens-shading 仍比 5 min 更干净，
> 20 min 提升边际明显减少。10 min 是 cost / quality 甜点。
> 详见 `docs/archived_recipes.txt` 第 5 节的 λ 扫描记录。

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素。
  子图分辨率减半，通道维不变。
- **L1 + λ·TV loss**（`n2n/trainer.py` 内联）：
  ```
  l1 = F.l1_loss(f(g₁), g₂)
  tv = mean(|∂x f(g₁)|) + mean(|∂y f(g₁)|)
  loss = l1 + tv_lambda * tv
  ```
  L1 比 L2 在中位数上更鲁棒；TV 仅在去噪输出上施加梯度惩罚，
  抑制 lens-shading 角落的低频 blob 残留。
- **残差 UNet** (`n2n/model.py`)：`denoise(x) = x − forward(x)`，
  网络学习"噪声残差"，输出端零初始化让训练从恒等映射出发，
  对 600s 这种短预算特别友好。
- **FP16**：`torch.amp.autocast` + 自动 dtype 选择（Ampere+ 走 bf16，Turing 走 fp16）。
- **GUI 与训练解耦**：训练跑在 `QThread`，通过 Qt signal 把 `StepInfo` /
  `PreviewInfo` 发回 UI 主线程。
- **GPU-resident dataset** (`n2n/gpu_dataset.py`)：140 帧 packed 一次性
  贴到 GPU（~744 MB），random crop + flip / transpose 全部 GPU 上做，
  消除 H2D copy 和 DataLoader 启动开销。
- **torch.compile(mode='reduce-overhead')**：4090 上 ~91 → ~165 sps（paper N2N）/
  ~150 → ~290 sps（L1+TV），主要是 CUDA Graph 把每 step kernel launch 开销吃掉。
  需要 `triton==3.1.0` 与 `torch 2.5.x` 配套。

## 5. 工程结构

```
new_denoise/
├── n2n/                              # 生产代码
│   ├── raw_utils.py                  # 16-bit display pipeline + I/O + Bayer pack
│   ├── n2n_sampler.py                # vectorised pair sampling
│   ├── dataset.py                    # CPU DataLoader fallback
│   ├── gpu_dataset.py                # GPU-resident 数据集（默认）
│   ├── model.py                      # 残差 UNet (max_pool + ConvTranspose2d)
│   ├── trainer.py                    # 训练循环：L1 + tv_lambda·TV
│   └── infer.py                      # tile 推理 + ImageJ-stack 输出
├── run_train.py                      # 一键启 GUI
├── train_gui.py                      # PyQt5 GUI 实现
├── benchmark.py                      # 1-3 分钟吞吐量基准
├── docs/
│   ├── EXPERIMENTS_JOURNEY.md        # 整个项目的来龙去脉 + 经验
│   ├── archived_recipes.txt          # 主干放弃过的方法（paper N2N / VGG / robust / 等）
│   └── how_to_compare.txt            # 怎么写一份对比 HTML 报告
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

## 6. 资源占用参考（RTX 4090 + bf16 + torch.compile）

| 配置 | step/sec | 显存 | 备注 |
|---|---|---|---|
| 残差 UNet + L1 + 0.01·TV (depth=3, patch=256, batch=6) | ~285 | ~2 GB | 单 forward / step |
| paper N2N（已存档）  L2 + γ·L_reg                     | ~163 | ~2 GB | 多一次 full-image forward (一致性正则) |

整图 1280×1088 推理 + 512 tile + 32 overlap，单帧约 0.05–0.45 s（首张含 cuDNN 算法选择开销）。
