# Quickstart · 在新机器上跑起来

把这个 zip 解压到任意位置，按下面的步骤来。

## 0. 前置条件

- **Windows / Linux 都行**
- **Python 3.10 / 3.11**（不要太旧，PyTorch AMP 需要）
- **NVIDIA GPU** + 安装好 CUDA-enabled PyTorch（开发环境是 RTX 4090，
  最低 4 GB 也能跑——把 `batch_size` 调小到 2 或 3）
- 训练数据放到 `train_data/Data/<场景>/<iso>/sensorN_raw.png`
  （`raw` 是 1280×1088 `uint16` Bayer-RGGB；`isp` 仅作视觉参考，不参与训练）

## 1. 解压 + 装依赖

```bash
unzip 8bit_denoise.zip -d 8bit_denoise
cd 8bit_denoise

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 里 `torch>=2.0`，pip 装的是 CPU 版的话需要手动安装 CUDA 版：

```bash
# 例如 CUDA 12.1
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

装完验证一下：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

期望输出类似 `cuda: True NVIDIA GeForce RTX 4090`。

## 2. 准备数据

```
train_data/
└── Data/
    ├── changqiao_qiaodi/
    │   └── 16/                       # ISO=16 这级（最噪的，自监督训练首选）
    │       ├── sensor2_raw.png       # 1280×1088 uint16 Bayer-RGGB
    │       ├── sensor2_isp.png       # 1280×1088 uint8 BGR (vendor ISP, 仅参考)
    │       ├── sensor3_raw.png
    │       └── sensor3_isp.png
    ├── langan_louti/16/...
    └── ...
```

每个场景至少准备 2 张 `*_raw.png`（一对 sensor2 / sensor3）。
Bayer 模式必须是 RGGB，黑电平约 ~9 in 0-255 域。

## 3. 跑 3 分钟 benchmark（推荐先做）

```bash
python benchmark.py
```

无参数。3 分钟内跑 N2N 默认配置（L2 + γ·L_reg），最后给出本机 step/sec
和外推到 10 / 20 min 的训练吞吐量。

## 4. 真正训练

GUI 版（推荐）：

```bash
python run_train.py 600         # 10 分钟（最低稳定值，推荐起点）
python run_train.py 1200        # 20 分钟
```

> ⚠️ **训练时长 ≥ 600 s**。非残差 UNet 没有零初始化的恒等先验，
> 5 min 以下有概率出现"灾难棋盘"（output 是大片彩色 2×2 网格）；
> 10 min 是稳定区。

GUI 顶部能调 `data root` / `train`（秒）/ `batch` / `patch` / `lr` / `FP16`。
点 `Start training`，看 loss 曲线（**rec L2 / reg L2 / total** 三条线，log-log）+
64×64 noisy/denoised 对比；训练结束后点 `Generate result/ comparison`，
全量 raw 推理结果会写到 `result/<scene>/<iso>/sensorN_compare.png`。

Headless 版：

```bash
python -c "from n2n.trainer import Trainer, TrainConfig; Trainer(TrainConfig()).run()"
```

默认会得到这些 ckpt：

```
checkpoints/
  n2n_model.pt          # 最终 (10 min)
  n2n_model_300s.pt     # 5 min 中间快照
  n2n_model_600s.pt     # 10 min 中间快照
```

## 5. 推理（不依赖 GUI）

```python
from n2n.infer import run_inference_set
run_inference_set(
    ckpt_path="checkpoints/n2n_model.pt",
    data_root="train_data/Data",
    out_root="result",
)
```

会对 `data_root` 下所有 `*_raw.png` 跑滑窗 tile 推理（512 tile + 32 overlap），
每张写出：

- `<sensor>_compare.png` —— 三联对比（noisy / denoised / vendor ISP）
- `<sensor>_isp_without_denoise.png` —— ImageJ stack 用，identical 尺寸 + 共享 WB
- `<sensor>_isp_with_denoise.png`
- `<sensor>_denoised_raw.png` —— 16-bit 去噪 raw

## 6. 看实验报告

整个项目所有失败和成功的实验来龙去脉在
[`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md)。
最重要的一个教训是：**Neighbor2Neighbor 论文的一致性正则项不能省**。
没这一项的训练等价于普通自监督，会出可见的 2×2 grid artifact，加任何
mode penalty / VGG perceptual loss 都只是给症状打补丁。

## 7. 常见问题

**Q：CUDA out of memory**
A：把 `batch` 从 6 改成 4 或 3 试试；最少 1。
patch_size 也可以从 256 降到 192 / 128 解决。

**Q：sps 比预期低**
A：先确认 `torch.cuda.is_available()` 返回 True；其次 `cudnn.benchmark = True`
（trainer 已经设了）。FP16 必须开（默认开），关掉会慢一倍。

**Q：训练 loss 不下降**
A：N2N 训练特点。看 `rec` 项（自监督下界 ≈ 噪声地板），不要看 `total`——
`reg` 项随 γ 增大而被放大，单条曲线很难判断收敛。GUI 的 log-log 图
把三条线画在一起就是为了直观看趋势。

**Q：训练结果出现彩色 2×2 棋盘**
A：训练时间不够。非残差 UNet 没有零初始化恒等先验，需要至少 10 min
才能稳定收敛。如果训练时长 ≥ 600 s 但仍然出现，检查
`TrainConfig.gamma_final` 是不是被改成 0 了——一致性正则项是消除
2×2 grid 的核心。

**Q：去噪后大平面有色阶 / 锯齿**
A：检查 `n2n/raw_utils.py` 是否完整（应该有 `raw_to_linear_bgr` 函数）。
旧版本在 demosaic 之前 `astype(uint8)` 会触发这个 bug。
