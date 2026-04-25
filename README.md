# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 16-bit RGGB Bayer raw（`train_data/Data/<scene>/<iso>/sensorN_raw.png`）做
**Neighbor2Neighbor** 自监督单帧去噪。残差预测、`L1 + 0.03·TV` 损失、
FP16，跑在 **RTX 2070** 上。带一个 PyQt5 GUI，实时显示 loss 曲线（log-log）、
64×64 patch 的 noisy / denoised 对比、训练进度，结束后一键把全量推理结果写入 `result/`。

> **完整实验过程和参数选择的来龙去脉在 [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)**，
> 含 23 个候选的扫描和 7 场景 20 ROI 视觉验证，包括为什么默认是 `lam=0.03` 训练 20 min。

## 0. 快速上手

新机器部署见 [`QUICKSTART.md`](QUICKSTART.md)。一行话：

```bash
pip install -r requirements.txt
python benchmark.py             # 跑 3 min 看本机 sps，估算合适训练时长
python run_train.py             # 启 GUI 真正训练
```

## 1. 数据规格

- raw：`uint16`, 1280×1088，单通道 Bayer，**RGGB** 模式
- isp：`uint8` BGR，与 raw 同尺寸，仅作视觉参考，**不参与训练**
- 7 个场景 × 10 个 ISO × 2 个 sensor = 140 对
- 网络输入：`raw / 65535` → `float`（FP16），4 通道打包形状 `(B, 4, H/2, W/2)`，
  通道顺序 `R, Gr, Gb, B`
- ISO=16 是最噪的（自监督训练通常在最高 ISO 上效果最好）

## 2. 显示流水线（避免 staircase 的关键）

raw → 显示，全程 16-bit / float32，最后才量化到 8-bit
（实现在 `n2n/raw_utils.py::raw_to_display`）：

```
raw_uint16
  -> 16-bit black-level subtract            (raw_to_linear_bgr)
  -> 16-bit demosaic (cv2 BayerRG2BGR)
  -> float32 BGR 0-255
  -> gray-world WB                          (raw_to_display)
  -> gamma 2.2
  -> uint8                                  (only here!)
```

之前的实现在 demosaic 之前就 `astype(uint8)` 了，导致大平面出现明显色阶。详见
`docs/EXPERIMENTS_JOURNEY.md` 阶段 1-2。

## 3. 训练

```bash
pip install -r requirements.txt
python run_train.py                   # 一键 GUI（推荐）
# 或者 headless：
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

### 默认配置（已被实验敲定）

`n2n/trainer.py` 的 `TrainConfig`：

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

会得到：

```
checkpoints/n2n_model.pt          # 最终 (20 min)
checkpoints/n2n_model_120s.pt     # 2 min 快照
checkpoints/n2n_model_300s.pt     # 5 min 快照
checkpoints/n2n_model_600s.pt     # 10 min 快照
```

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素，
  整批共享同一 mask。子图分辨率减半，通道维不变。
- **L1 + 0.03·TV 损失**（`n2n/losses.py`）：
  ```
  main = L1(g1 - f(g1), g2) + 0.03 * TV(g1 - f(g1))
  ```
  TV 项压平大平面（暗角彩噪），L1 主项保留细节。`losses.py` 还留了 7 个其他损失
  方便日后实验，但默认只用 `l1_plus_tv`。
- **残差头零初始化**：`UNet.out_conv` 权重 / 偏置全 0，网络起始等价于恒等，训练稳定。
- **FP16**：`torch.amp.autocast` + `GradScaler`，矩阵运算 fp16。
- **按时间停止 + 中间快照**：trainer 每 epoch 末检查 `pending_saves`，
  到点就在主 ckpt 旁存一份 `_<seconds>s.pt`。
- **GUI 与训练解耦**：训练跑在 `QThread`，通过 Qt signal 把 `StepInfo` /
  `PreviewInfo` 发回 UI 主线程。

## 5. 工程结构

```
8bit_denoise/
├── n2n/                         # 生产代码
│   ├── raw_utils.py             # 16-bit 显示流水线、I/O、Bayer pack
│   ├── n2n_sampler.py           # 矢量化 N2N 子采样
│   ├── dataset.py               # patch + flip/transpose 增强
│   ├── model.py                 # 可配置深度 UNet (residual head)
│   ├── losses.py                # 8 个 loss（默认 l1_plus_tv lam=0.03）
│   ├── trainer.py               # FP16 训练循环 + 中间 ckpt
│   └── infer.py                 # tile 推理 + ImageJ-stack 输出
├── train_gui.py                 # PyQt5 GUI
├── run_train.py                 # 一键启动 GUI
├── benchmark.py                 # 3 分钟吞吐量基准（部署到新机器先跑这个）
├── docs/                        # 实验过程和知识沉淀
│   ├── EXPERIMENTS_JOURNEY.md   # ← 看这个！整个项目的故事和经验
│   ├── REPORT.html              # 16 策略初步排名
│   ├── TV_LAMBDA_REPORT.html    # 4 个 λ 横向比较
│   ├── TV_LAM01_5VS10_REPORT.html
│   ├── TV_LAMBDA_DURATION_REPORT.html
│   ├── TV_20ROI_REPORT.html     # 7 场景 20 ROI 视觉判定
│   └── TV_LAM03_DURATION_REPORT.html  # 当前默认 lam=0.03 的 4 时长对比
├── requirements.txt
├── README.md                    # （本文件）
├── QUICKSTART.md                # 部署到新机器的步骤
├── train_data/                  # 训练数据（自备，gitignore）
├── checkpoints/                 # 训练好的 .pt（自动建，gitignore）
└── result/                      # 推理对比图（自动建，gitignore）
```

## 6. 资源占用参考（RTX 2070 + FP16）

| 配置 | step/sec | 显存 | 备注 |
|---|---|---|---|
| 默认 (depth=3, patch=256, batch=6) | ~36.4 | ~5.5 GB | 1 epoch ≈ 4.7 s |
| depth=4, patch=256, batch=4 | ~22 | ~7.5 GB | 大网络对比 |

整图 1280×1088 推理 + 512 tile + 32 overlap，单帧约 0.3 s。
