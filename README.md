# 8-bit Bayer 去噪 · Neighbor2Neighbor (PyTorch + PyQt5)

针对 `train_data/Data/<scene>/<iso>/sensorN_raw.png` 这批 16-bit RGGB Bayer 噪声图，
用 **Neighbor2Neighbor**（自监督，单帧噪声）训练去噪网络，残差预测、L1 loss、FP16，
跑在 **RTX 2070** 上。带一个 PyQt5 GUI，实时显示 loss 曲线（log-log）、
64×64 patch 的 noisy / denoised 对比、训练进度，结束后一键把全量推理结果写入 `result/`。

## 1. 数据规格（已验证）

- raw：`uint16`, 1280×1088，单通道 Bayer，**RGGB** 模式（已通过对比 ISP 参考图确认：
  RGGB 解码出来的木凳是棕色，与 ISP 参考一致；其他三种模式色彩明显错乱）
- isp：`uint8` BGR，与 raw 同尺寸，仅供视觉参考 / 训练后对比，**不参与训练**
- 共 7 个场景 × 10 个 ISO 等级 × 2 个 sensor = 140 对
- 网络输入：`raw / 65535` → `float`（FP16），4 通道打包后形状 `(B, 4, H/2, W/2)`，
  通道顺序 `R, Gr, Gb, B`

## 2. 显示流水线（用户规定）

raw → 显示的固定步骤（实现于 `n2n/raw_utils.py::raw_to_display`）：

1. `img8 = raw_uint16 / 256.0`，转到 0-255 域
2. `img8 - 9`（0-255 域的 black level）
3. `cv2.cvtColor(..., COLOR_BayerRG2BGR)` 去马赛克
4. `pow(img/255, 1/2.2) * 255` gamma 2.2 显示

## 3. 训练

```bash
pip install -r requirements.txt
python train_gui.py
```

GUI 顶部可以设置：
- `data root`、`train`（训练时长，秒）、`batch`、`patch`（4 通道域 patch，默认 128 → Bayer 256×256）、
  `lr`、`FP16` 开关
- `Start training` 启动；`Stop` 在当前 step 结束后停下
- 训练中：loss 曲线（log-log）实时更新；64×64 patch（4 通道域）通过显示流水线
  转 RGB，再用 nearest-neighbour 放大 16×（≈1024×1024）显示左右 noisy / denoised 对比
- 训练时间到 → 进度条 100% → 状态栏显示 **TRAINING FINISHED**
- 然后点 `Generate result/ comparison`：对全量 140 张 raw 推理，并把
  `noisy raw → ISP-style | denoised raw → ISP-style | vendor ISP` 三联拼接图
  写入 `result/<scene>/<iso>/sensorN_compare.png`，同时把去噪后的 16-bit raw 存为
  `sensorN_denoised_raw.png`

## 4. 关键实现要点

- **Neighbor2Neighbor sampler** (`n2n/n2n_sampler.py`)：每个 2×2 cell 随机选一对相邻像素，
  一个进 g1、另一个进 g2，整批共享同一 mask。子图分辨率减半，但通道维不变。
- **N2N loss**（`trainer.py`）：
  ```
  main = L1(g1 - f(g1), g2)                # f 输出预测噪声残差，input - f(input) = denoised
  reg  = L1((g1 - f(g1)) - g2, g1(input - f(input)) - g2(input - f(input)))
  loss = main + λ * reg                    # 默认 λ = 1.0
  ```
  按用户要求只用 L1 loss，主项 + 一致性正则。
- **残差训练**：`UNet.out_conv` 权重/偏置全初始化为 0，所以网络起始等价于恒等映射，训练稳定。
- **FP16**：`torch.amp.autocast('cuda')` + `GradScaler('cuda')`，所有矩阵运算 fp16，
  scaler 自动处理梯度缩放。
- **按时间停止**：训练循环每个 step 检查 `time.time() - t0 >= cfg.train_seconds`，到了就跳出，
  保存 checkpoint，发出 `finished_signal`，UI 显示训练结束。
- **GUI 与训练解耦**：训练在 `QThread` worker 中运行，通过 Qt signal 把
  `StepInfo` / `PreviewInfo` 发回主线程刷新 UI，避免冻结。

## 5. 工程结构

```
8bit_denoise/
├── train_data/Data/...          # 已有数据
├── n2n/
│   ├── __init__.py
│   ├── raw_utils.py             # I/O、4 通道打包、显示流水线、Bayer pattern
│   ├── n2n_sampler.py           # Neighbor2Neighbor 邻居子采样
│   ├── dataset.py               # 随机 crop、增强（保留 Bayer）、缓存
│   ├── model.py                 # 小 UNet，残差头零初始化
│   ├── trainer.py               # 训练循环 + 回调（loss / preview / finish）
│   └── infer.py                 # 整图 tile 推理 + 三联对比拼图
├── train_gui.py                 # PyQt5 GUI 入口
├── requirements.txt
├── checkpoints/                 # 训练好的 .pt（自动建）
└── result/                      # 最终对比图（自动建）
```

## 6. 资源占用参考

- patch_size 128（4 通道域，对应 Bayer 256×256）、batch 4、`base_channels=48` UNet、FP16：
  RTX 2070 上单 step ≈ 25 ms，显存 ≈ 3.5 GB
- 推理整图 1280×1088（4 通道域 640×544）用 512 tile + 32 overlap，单帧 ≈ 0.3 s
