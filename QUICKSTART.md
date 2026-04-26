# Quickstart · 在新机器上跑起来

把这个 zip 解压到任意位置，按下面的步骤来。

## 0. 前置条件

- **Windows / Linux 都行**，开发用的是 Windows + PowerShell
- **Python 3.10 / 3.11**（不要太旧，PyTorch AMP 需要）
- **NVIDIA GPU** + 安装好 CUDA-enabled PyTorch（开发环境是 RTX 2070 8 GB，
  最低 4 GB 也能跑——把 `batch_size` 调小到 2 或 3）
- 训练数据放到 `train_data/Data/<场景>/<iso>/sensorN_raw.png`
  （`raw` 是 1280×1088 `uint16` Bayer-RGGB；`isp` 仅作视觉参考，不参与训练）

## 1. 解压 + 装依赖

```bash
# 解压（任选一种）
unzip 8bit_denoise.zip -d 8bit_denoise
# 或: tar -xzf 8bit_denoise.tar.gz

cd 8bit_denoise

# 强烈建议用 venv，避免污染系统 Python
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

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

期望输出类似 `cuda: True NVIDIA GeForce RTX 2070`。

## 2. 准备数据

把数据按下面的目录结构放好：

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

## 3. 跑 3 分钟 benchmark（强烈推荐先做这步）

```bash
python benchmark.py
```

无参数，直接 run。会做的事：

1. 用当前 `TrainConfig` 默认值（depth=3, patch=256, batch=6, FP16）
2. 跑 3 分钟，每 5 个 epoch 打一行进度
3. 最后报告本机的 **step/sec**、3 min 跑到第几 epoch
4. 把 sps 外推到 5 / 10 / 20 min 的 epoch 数
5. 给出推荐训练时长（让你的机器达到和 dev box 同样的 258 epoch）

参考输出（开发机 RTX 2070）：

```
=== benchmark · l1_plus_tv({'lam': 0.03}) · patch=256 batch=6 depth=3 ===
  duration : 180s (3.0 min)

   ep=  5  step=  850  t=  27s  sps=37.0
   ep= 10  step= 1700  t=  50s  sps=38.1
   ...
   ep= 35  step= 5950  t= 163s  sps=36.8

=== summary ===
  wall-clock     : 180.0 s
  steps          : 6 552
  epochs         : 38
  steps / sec    : 36.4
  steps / epoch  : 170

  projected epochs : 5 min -> 64   10 min -> 129   20 min -> 257

  reference: dev box (RTX 2070) reaches 258 epochs at 20 min.
  to match the same epoch count on THIS machine:
     train_seconds = 1204 s (20.1 min)
```

如果你的机器更慢（比如 1660Ti），benchmark 会算出更长的推荐时长，
你只要把 `n2n/trainer.py` 里的 `train_seconds` 改成那个值就行。

## 4. 真正训练

GUI 版：

```bash
python run_train.py
```

GUI 顶部能调 `data root` / `train`（秒）/ `batch` / `patch` / `lr` / `FP16`。
点 `Start training`，看 loss 曲线 + 64×64 noisy/denoised 对比；
训练结束后点 `Generate result/ comparison`，全量 raw 推理结果会写到
`result/<scene>/<iso>/sensorN_compare.png`。

Headless 版：

```bash
python -c "from n2n.trainer import Trainer, TrainConfig; Trainer(TrainConfig()).run()"
```

得到 4 个 ckpt：

```
checkpoints/
  n2n_model.pt          # 最终 (20 min)
  n2n_model_120s.pt     # 2 min 快照
  n2n_model_300s.pt     # 5 min
  n2n_model_600s.pt     # 10 min
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

整个项目（4 阶段：初始扫描 → TV 调优 → 7h 92 候选 → 3h 鲁棒 →
6 候选感知损失 / VGG ★）的全部来龙去脉、量化结果和经验总结都在
`docs/EXPERIMENTS_JOURNEY.md`，markdown，任意编辑器 / GitHub 直接看。

历史的 HTML 报告（REPORT.html / TV_*.html / RESEARCH_7H_REPORT /
REPORT_3H_ROBUST / REPORT_PERCEPTUAL）作为产出过程的细节存档，
已经放到本地 `_archive_local/docs_html/`（gitignored），需要的时候
还可以双击打开。

## 7. ★ VGG 推荐方案

如果想跑当前认为视觉效果最好的方案（残留噪声看起来"自然"），一行话：

```bash
python train_vgg.py
```

会顺序训出两个 ckpt 到 `ckpts/`：

- `ckpts/P0_baseline.pt` — 同款 robust 训练 (EMA + R2R + mask + jitter) + L1+0.03·TV（对照）
- `ckpts/P4_vgg.pt`      — 同款 robust 训练 + VGG-16 feature L1 ★

**推理路径与 baseline 完全一致**（用 `n2n.infer.run_inference_set`），
BPU / ONNX / TRT 部署 0 改动——VGG 只在训练时走一次 frozen forward。
配置 / 原理见 `docs/EXPERIMENTS_JOURNEY.md` 阶段 6。

## 8. 常见问题

**Q：CUDA out of memory**
A：`run_train.py` 启动 GUI 后，把 `batch` 从 6 改成 4 或 3 试试；最少 1。
patch_size 也可以从 256 降到 192 / 128 解决。

**Q：sps 比 dev box 低很多怎么办**
A：先确认 `torch.cuda.is_available()` 返回 True；其次 `cudnn.benchmark = True`
（trainer 已经设了）。FP16 必须开（默认开），关掉会慢一倍。

**Q：训练 loss 不下降**
A：N2N 训练特点。看 `main_loss` 而不是总 loss——总 loss 包含 TV 正则项
会随模型收敛而上涨。详见 `EXPERIMENTS_JOURNEY.md` 经验 #3。

**Q：去噪后大平面有色阶 / 锯齿**
A：检查 `n2n/raw_utils.py` 是否完整（应该有 `raw_to_linear_bgr` 函数）。
旧版本在 demosaic 之前 `astype(uint8)` 会触发这个 bug。

**Q：想换 lam 或换损失重新扫描**
A：改 `n2n/trainer.py` 里 `TrainConfig.loss_kwargs = {"lam": 你想要的}`
然后 `python run_train.py` 即可。所有历史扫描脚本（16 策略 / 92 候选 /
6 感知损失候选）都搬到 `_archive_local/` 本地保留，参考
`docs/EXPERIMENTS_JOURNEY.md` 第 7 节最末"已删除候选的配置存档"
能直接拿到当年的 loss_name + loss_kwargs。
