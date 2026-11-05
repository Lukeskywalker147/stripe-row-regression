# 线激光光条中心提取：逐行回归网络（v3）

场景无关的线激光光条中心提取。网络对每一行图像在 x 方向上做 softmax（W 个像素 + 一个"该行无光条"类别），再用 soft-argmax 和局部质心得到亚像素中心。本仓库包含代码、训练用的代表性数据、训练日志、权重（Release 附件）和评测结果。

> 状态：研究原型。没有可信的真实世界真值，下文所有结论都标注了依据和局限。

## 1. 方法概要

- 模型：`segmentation_models_pytorch` 的 U-Net（resnet18，从零训练），4 个输出通道（行头 + 列头），12.46M 参数。1080p 下约 8.6 ms/帧（GPU，bf16，v2 实测）。
- 解码：取每行前 3 个峰，±3 像素质心，Viterbi 平滑；`OrientRegressor` 按峰值得分自动选择条纹方向。
- 损失：对以真值为中心、σ=1 的高斯做交叉熵，加上 ±3 窗口内 soft-argmax 的 0.5·L1。
- 训练数据：约 50% 合成、50% 真实木头帧。
  - 合成：在线生成，随机激光颜色、宽度（0.8–5.5 px）、幅度（30–900）、程序化或真实（修复后的木头）背景、最多 4 条干扰条纹、高光、遮挡、8% 空图。标签精确（x 坐标 / -1 无光条 / NaN 忽略）。CPU 版 `synth3.py`，GPU 版 `synth_gpu.py` + `core3g.py`。
  - 真实：`data/real_train/data_scan_v3`（598 训练 + 120 验证，全是木头），标签取自 Steger 与 mask 一致的行，因此标签精度受这些方法限制（循环依赖）。

## 2. 目录

```
code/colreg/      训练、推理、合成、评测脚本（train3.py / train3g.py / infer.py / synth*.py / bench_*.py …）
code/misc/        早期分割阶段的辅助脚本
code/board_eval/  标定板上的评测脚本（直线 / 平面残差）
data/real_train/  data_scan_v3（真实帧 + mask）、bg（150 张修复背景）、labels（行标签 npz）、hand_labels
data/eval/        cam_snaps（16 张塑料头盔 / 水杯）、board25（25 张激光打在 ChArUco 板上的图）
logs/             每个 run 的 log.json / meta.json 和训练日志
results/          基准、对比图
docs/             阶段性报告（report.md，含较早的结果，部分已被取代）
release_assets/   权重（11 个 .pt，每个约 48 MB）和 SHA256SUMS.txt —— 不在 git 里，见第 6 节
setup.sh          把数据链接到脚本期望的位置并摆好权重
```

## 3. 复现

环境：Python、PyTorch（CUDA）、torchvision、`segmentation-models-pytorch`、OpenCV、NumPy。

```bash
./setup.sh                      # 链接数据，把 release_assets/*.pt 放到 code/colreg/runs/<name>/last.pt
cd code/colreg
python train3.py  --name r18_v3  --iters 16000          # CPU 合成（约 27 min / 单卡）
python train3g.py --name r18_v3g --iters 16000 --gpu_synth   # GPU 合成
python bench_eval.py --nosteger --models name=runs/r18_v3/last.pt:orient
```

主要参数见 `train3g.py`：`--bs 8 --lr 1e-3 --p_synth 0.5 --p_rot 0.5 --seed 0`。消融设置见 `run_abl.sh`（每个 8000 次迭代）。部分脚本里的路径是作者环境的 `~/stripe_seg/...`，需要按实际位置调整。

推理：

```python
from infer import OrientRegressor
m = OrientRegressor("runs/r18_v3/last.pt")
pts = m.extract(bgr_image)[:, :2]    # N x 2, (x, y) 亚像素
```

## 4. 结果与可靠度

### 合成基准（12 个难度轴 × 5 级，固定种子）
- v3 修复了颜色（红光漏检 23%→0）、低对比度（幅度 30 漏检 100%→0）、方向（旋转后 v2 约 85% 漏检→0）。
- 残余问题：干扰条纹（约 8% 假点，约定"最强条纹为目标"）、遮挡（约 7% 假点）。
- 局限：基准与训练生成器同分布，收益会被高估。

### 消融（8000 次迭代；基准每级 20 张 + 7 帧手工标注）
- 预训练、全分辨率跳连、纯合成：无超出种子噪声的收益。
- 分割头更差（空图平均 50 个假点/张）；纯真实数据明显最差；16000 次迭代使干扰条纹假点减半。
- GPU 合成数据（v3g）与 CPU 版（v3）验证集一致，基准上干扰条纹略差（12.6% 对 7.9%），各只有一个种子，不能说等价。
- 手工标注只有 7 帧，点击噪声 2–3 px，覆盖率指标不惩罚假点。

### 激光打在 ChArUco 标定板上（25 帧，`code/board_eval`）
黑白格交替使光条不连续。NN 的中心点反投影到 PnP 标定板平面后，与 Steger 比较：

| | Steger | v3 | v3g |
|---|---|---|---|
| 落在直线上的点占比 | 34% | 99% | 99% |
| 每帧有效点数 | 131 | 476 | 475 |
| 直线残差 RMS (mm) | 0.91 | 0.31 | 0.33 |
| 平面拟合 RMS (mm) | 1.01 | 0.43 | 0.44 |

- 用 20 帧拟合激光平面、在另 5 帧上测，NN 的留出 RMS 为 0.45 mm，与全量拟合接近。
- NN 平面与 Steger 平面法向差 0.14°，质心处平移差约 0.5 mm；哪个更接近真实平面没有真值可判断。
- Steger 在这里用的是标定包里按蓝光调的参数，未针对这批图重调，3 倍差距里有一部分来自基线没调好。
- 直线残差偏爱平滑输出，也看不出整体偏移（例如落在光晕而不是光芯上）。
- 样本为 25 帧、单相机、单个种子；训练里没有棋盘格。

### 木球扫描（V03_150，37 帧，球面拟合残差）
Steger 0.43 mm，NN v3 0.37 mm（拟合半径 49.65 / 49.76 mm）。差距接近噪声，未做多种子检验。球面拟合会吸收整体平移和尺度误差，所以不能代表绝对精度。

## 5. 已知局限

- 没有真实世界真值；训练标签、辅助评测多处共享同一规则，存在循环。
- 真实训练帧只有木头；塑料件（`cam_snaps`）只用于评测。
- 金属、强镜面场景未测。
- 激光平面标定的真实收益需要用新平面去测量已知尺寸物体才能确认，尚未做。
- 手眼标定验证误差（约 2 mm）和时间同步等误差不在本工作范围内，可能盖过光条提取的改进。

## 6. 权重

11 个 `.pt`（v3 / v3g 为主，其余为消融和旧版本）放在 GitHub Release 附件里，校验值见 `release_assets/SHA256SUMS.txt`。下载后放入 `release_assets/` 再运行 `./setup.sh`。

| 文件 | 说明 |
|---|---|
| r18_v3.pt | 主模型，CPU 合成，16k it |
| r18_v3g.pt | GPU 合成，16k it |
| r18_v2.pt | v2，12k it，需竖直条纹 |
| r18_scratch_s0.pt | v1，6k it |
| abl_*.pt | 消融（8k it）：base_s0/s1、pre、seg、fullres、synth、real |

## 7. 数据说明

`data/` 下是作者自己采集的木头扫描帧、塑料件照片和标定板图像。使用或再分发前请确认你有权这样做。
