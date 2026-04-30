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

## 3. 跑 1 分钟 speedtest（推荐先做）

```bash
python speedtest.py
```

无参数，默认 1 分钟。跑当前主干 L1 + 0.01·TV，过滤掉前 30 s 的
`torch.compile` 首编译期，报告 **稳态 sps**（warmup 后 epoch 的中位数）+
外推到 5 / 10 / 20 min 的训练吞吐量。整体平均 sps 在 1 min 测试里会被
~30 s 的编译期严重拉低，所以这里直接给稳态。

## 4. 真正训练

GUI 版（推荐）：

```bash
python run_train.py 600         # 10 分钟（baseline，cost/quality 甜点）
python run_train.py 1200        # 20 分钟（视觉略好，loss 几乎不变）
```

> 5 min 时 loss 已经基本收敛（~0.0162），但视觉上 10 min 在 ISO=16
> 暗区 / 角落 lens-shading 仍更干净，20 min 边际收益较小。10 min 是
> baseline，详见 `docs/EXPERIMENTS_JOURNEY.md` 阶段 8 + 附录 A.6（λ 扫描）。
> ⚠ 在新对称-ISP benchmark 下"训得更长"边际趋零，要把 score 压低主要靠
> ensemble / TTA（详见阶段 10）。

GUI 顶部能调 `data root` / `train`（秒）/ `batch` / `patch` / `lr` / `FP16`。
点 `Start training`，看 loss 曲线（**L1 + λ·TV** 单条，log-log）+
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

- [`docs/EXPERIMENTS_JOURNEY.md`](docs/EXPERIMENTS_JOURNEY.md) — 整个项目
  所有失败 / 成功实验的连续叙事（16-strategy sweep / 92-candidate grid /
  鲁棒推 / 6-candidate perceptual / **阶段 10 新 benchmark 攻坚**）。
  附录 A 是已废弃方法（paper N2N、L1+VGG、robust_trainer、multi-scale 等）
  的 recipe 快照 + git 复活步骤。
- [`docs/how_to_compare.md`](docs/how_to_compare.md) — 怎么写一份对比
  HTML 报告（多 ckpt 推理、ROI 切片、HTML 模板风格）。
- [`docs/how_to_run_benchmark.md`](docs/how_to_run_benchmark.md) — ckpt → ONNX
  → benchmark 的完整流程（旧 ysstereo 流程的历史复盘 + 仍然适用的 ONNX 合同）。

## 7. 常见问题

**Q：CUDA out of memory**
A：把 `batch` 从 6 改成 4 或 3 试试；最少 1。
patch_size 也可以从 256 降到 192 / 128 解决。

**Q：sps 比预期低**
A：先确认 `torch.cuda.is_available()` 返回 True；其次 `cudnn.benchmark = True`
（trainer 已经设了）。FP16 必须开（默认开），关掉会慢一倍。
torch.compile 也很关键：4090 上从 ~150 sps 跳到 ~290 sps，需要
`triton==3.1.0` 与 `torch 2.5.x` 配套。

**Q：训练 loss 不下降**
A：L1+TV 在 RTX 4090 上 ~5 min 时 loss 已基本平台（~0.0162），后面
几乎不再下降。**loss 平台 ≠ 视觉到顶**——10 min 在暗区 / lens-shading
角落比 5 min 明显更干净，20 min 仍有微小提升。看 result/ 里的全图
对比比看 loss 数字更靠谱。

**Q：去噪后大平面有色阶 / 锯齿**
A：检查 `n2n/raw_utils.py` 是否完整（应该有 `raw_to_linear_bgr` 函数）。
旧版本在 demosaic 之前 `astype(uint8)` 会触发这个 bug。

**Q：去噪后 lens-shading 角落仍有低频 blob**
A：`tv_lambda` 从 0.01 调到 0.02 / 0.03 试试（强 TV → 平坦区干净，
但代价是细节被吃掉，所以要看 ROI 视觉效果决定）。详见
`docs/EXPERIMENTS_JOURNEY.md` 附录 A.6 的 λ 历史扫描。
