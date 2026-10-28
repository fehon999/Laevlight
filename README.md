# EvLight 单阶段训练项目

本项目使用 EvLight 低照度图像–事件融合主干和参考帧特征分支。原来的 `pretrain_all` 现为完整训练过程：所有网络参数由同一个优化器联合更新，训练结束即可直接推理。网络的构造、初始化和前向计算保持不变。

## 四个数据集独立配置

每份 YAML 都包含完整的 `data`、`loader`、`model`、`loss`、`train` 和 `inference` 设置，可以分别修改，不需要组合其他配置。

| 数据集 | 配置文件 | 数据读取器 | 逐帧事件所在目录 | 训练输出 |
|---|---|---|---|---|
| SDE 室内 | `configs/sde_in.yaml` | EGLOL | `low/` | `outputs/sde_in/` |
| SDE 室外 | `configs/sde_out.yaml` | EGLOL | `low/` | `outputs/sde_out/` |
| SDSD 室内 | `configs/sdsd_in.yaml` | SDSD，`is_indoor: true` | `low/` | `outputs/sdsd_in/` |
| SDSD 室外 | `configs/sdsd_out.yaml` | SDSD，`is_indoor: false` | `normal/` | `outputs/sdsd_out/` |

首先把各配置的 `data.root` 改成实际数据根目录。当前 `data/sde_in` 等值是占位路径，根目录应直接包含 `train/` 和 `test/`。从项目根目录运行下面的命令；数据路径可以使用绝对路径。

```text
实际数据根目录/
  train/
    sequence_name/
      low/
        *.png                 # 低光图像
        *.npz                 # SDE / SDSD 室内逐帧事件
      normal/
        *.png                 # 正常光 GT / 参考图像
        *.npz                 # SDSD 室外逐帧事件
  test/
    sequence_name/
      low/
      normal/
```

四份配置均使用 `is_split_event: true`。事件文件中的 `arr_0` 为按时间升序排列的 `[N, 4]` 数组，列顺序为时间、x、y、极性；图像和对应事件按文件名排序配对。SDSD 延续原读取流程，对帧和逐帧事件列表执行 `[3:-2]` 截取。训练只读取 `train_split` 和 `test_split`，无需额外数据划分。

## 安装与训练

在具有合适 PyTorch 环境的终端安装依赖：

```bash
python -m pip install -r requirements.txt
```

根据数据集选择一条命令：

```bash
python train_event_dunet.py --config configs/sde_in.yaml
python train_event_dunet.py --config configs/sde_out.yaml
python train_event_dunet.py --config configs/sdsd_in.yaml
python train_event_dunet.py --config configs/sdsd_out.yaml
```

`--config` 必须显式指定。四份配置先沿用修改前的预训练设置，数据路径、读取协议和输出目录按数据集区分；具体超参数可以在对应配置中独立调整。

| 设置 | 当前值 |
|---|---|
| 训练轮数 | `train.epochs: 50` |
| 优化器 | AdamW，全部模型参数共同更新 |
| 初始学习率 | `train.lr: 0.00015` |
| 学习率调度 | cosine，每轮结束后更新，最低 `0.00000001` |
| 权重衰减 | `train.weight_decay: 0` |
| 训练批次 / 工作进程 | `8 / 8`，`drop_last: true` |
| 测试批次 | `1` |
| 图像裁剪 / 事件通道 | `256×256 / 32` |
| 混合精度 | `train.amp: true`，仅 CUDA 启用 |
| 损失 | `Charbonnier + 0.5 × LPIPS`，`ssim_weight: 0` |

每个训练批次进行一次前向、一次反向传播和一次优化器更新。每轮完成后测试并保存模型，然后更新学习率。训练结束再测试一次，保存最后一轮权重。

各数据集输出目录包含：

- `pretrain_all_latest.pt`：最近一轮权重与训练轮数、配置、测试结果。
- `best.pt`：测试集平均 SSIM 最高的一轮权重，记录 `best_ssim`；分数相同时保留较早的一轮。
- `global_last.pt`：全部训练结束后的权重与最终测试结果。

这些文件名沿用原预训练流程。`best.pt` 按测试集平均 SSIM 选取，PSNR 和损失仅作记录；因此该测试集参与了模型选择，不能将其当作完全独立的最终评估集。

初始化已有权重时使用：

```bash
python train_event_dunet.py --config configs/sde_in.yaml --resume outputs/sde_in/best.pt
```

`--resume` 仅加载模型权重；训练轮数、优化器和学习率调度从第一轮重新开始。完整参考分支权重需要匹配原来的网络配置；`resume_strict: false` 可用于只有 EvLight 主干的权重，但不能跳过同名参数的形状冲突。

## 推理

推理和训练使用同一份数据集配置，其中 `inference.checkpoint` 默认指向该数据集自己的 `best.pt`：

```bash
python infer_event_dunet.py --config configs/sde_in.yaml
python infer_event_dunet.py --config configs/sde_out.yaml
python infer_event_dunet.py --config configs/sdsd_in.yaml
python infer_event_dunet.py --config configs/sdsd_out.yaml
```

推理只加载所选数据划分。默认读取 `test/`；可以覆盖权重、划分目录和输出目录：

```bash
python infer_event_dunet.py --config configs/sde_in.yaml --checkpoint outputs/sde_in/global_last.pt --split test --output-dir outputs/sde_in/final_inference
```

`--split` 指定 `data.root` 下的目录名。`--save-gt` 额外保存 GT，`--strict` 强制严格加载权重；四份配置默认已设 `inference.strict: true`。

当 `--split` 为 `train` 或 `test` 时，分别采用配置中的 `data.train_split` 或 `data.test_split`；其他值直接作为划分目录名。只准备测试目录也可以推理。

## 参考分支与数据处理

四份配置均使用 `ReferenceBranch.enabled: true`、`fusion_mode: reference_affine`、`encoder_blocks: 2` 和 `residual_scale: 0.1`。主干特征只在进入 `Unet_ReFormer` 前接受一次参考残差注入。最终投影零初始化，初始注入为零；当前帧编码器与参考帧编码器随后共同参与训练。

推理中的参考仿射诊断只适用于 `reference_affine`。关闭参考分支或使用其他已有融合模式时，会自动跳过诊断，正常保存预测结果。

训练参考来自同一序列的其他正常光帧，排除当前帧 GT。当前配置 `reference_first_probability: 0.1`、`reference_dropout_probability: 0`；测试固定选择首个可用的非当前帧。没有可用参考时使用零图，并通过 `reference_valid: false` 关闭注入。

训练保留随机裁剪及同步水平翻转、垂直翻转，方形裁剪还可转置；低光图、GT、参考、模糊图、光照先验和事件采用同一空间变换。测试和推理保留中心裁剪，不做随机增强。

项目说明：

- [参考分支结构](docs/reference_branch.md)
- [网络与训练数据流](docs/event_guided_dunet_flow.md)
- [代码审阅与迁移说明](docs/pretrain_refactor.md)

## 检查代码

安装项目依赖后运行：

```bash
python -m unittest discover -s tests -v
```

当前 35 项测试通过，包括四类数据协议的合成数据训练与推理。旧版与新版的 CPU 数值对照、验证范围和未运行的真实数据实验见[代码审阅与迁移说明](docs/pretrain_refactor.md#验证结果)。
