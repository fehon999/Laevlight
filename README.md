# EvLight reference-frame adapter

本仓库使用 EvLight 作为低照度图像–事件融合 backbone，并增加了一个可独立开关的参考帧特征分支：

```text
low RGB + events -> EvLight preprocessing/fusion -> holistic feature --------+
light-up RGB -> current encoder -> channel mean/std --------+                 +--> ReFormer -> pred
reference RGB -> reference encoder -> channel mean/std -----+-> single gate + bounded affine-+
```

参考分支只在 `Unet_ReFormer` 之前做一次残差注入，不改 EvLight 的编码器、SNR 区域选择或解码器结构。最终投影使用零初始化，因此新分支刚加入预训练 EvLight 时，网络输出与未加入分支时相同。

## 训练

```bash
python train_event_dunet.py --config configs/event_mini_dunet_train.yaml
```

首先修改配置中的数据路径。数据目录格式为：

```text
root/
  train|validate|test/
    sequence_name/
      low/
        *.png
        *.npz
      normal/
        *.png
```

每个样本包含：

```text
low, event, gt, img_refer, reference_valid, ill_list
```

训练时从同一序列随机选择非当前帧的正常光图像作为参考；验证和测试固定选择首个可用的非当前帧。参考丢弃样本使用全零图，并通过 `reference_valid=false` 完全关闭注入，避免分支偏置影响输出。

训练和微调配置默认开启 `data.geometric_augmentation: true`，与 `evlight_refer` 示例一致：随机裁剪后，以各 50% 概率执行水平翻转、垂直翻转，方形裁剪另以 50% 概率转置。低光图、GT、参考帧、模糊图、光照先验和事件体素使用同一空间变换，事件的时间通道和极性不变。非方形裁剪不转置，避免批次尺寸不一致；query、验证和测试保持中心裁剪，不做随机增强。设为 `false` 可关闭新增增强；带 seed 的读取接口可复现增强。

默认配置使用 **80 轮单阶段联合训练**，主干和参考分支由同一个优化器、同一次反向传播更新，不冻结分支，不执行双层训练（`bilevel_reference_outer.epochs: 0`）。`train.stages.pretrain_all.lr` 为 `0.0002`，`min_lr` 为 `0.00000001`，每轮结束后更新 cosine 学习率，`T_max` 跟随 `epochs`。训练日志显示当轮实际学习率。将 `lr_scheduler` 改为 `constant` 可使用固定学习率。

训练目标为 `Charbonnier + 0.5 * LPIPS + 0.1 * (1 - SSIM)`。新增的 `loss.ssim_weight` 默认为 0（兼容旧配置），当前训练配置设为 0.1；SSIM 损失在关闭 autocast 的 FP32 中计算，使用与当前指标一致的 RGB、11×11 窗口和范围。不会修改测试指标或使用 GT 做亮度校正。是否改善 PSNR/SSIM 必须通过实验判断。

新实验输出保存到 `outputs/event_mini_dunet_affine_ssim`，推理配置同步使用该目录。`best.pt` 仍按现有流程中的最高测试 PSNR 保存，并非最高 SSIM；正式对比应使用独立验证集选模型，测试集只用于最终报告。

当前 `--resume` 仅加载模型权重，学习率调度会从第一轮重新开始，不恢复优化器或调度器进度。

## 参考分支配置与消融

```yaml
model:
  ReferenceBranch:
    enabled: true
    encoder_blocks: 2
    fusion_mode: reference_affine
    residual_scale: 0.1
    statistics_eps: 0.000001
    zero_init: true
```

`reference_affine` 根据当前帧和参考帧的特征均值/标准差预测通道缩放和偏移，经 `tanh` 限幅后，用一次门控修正主干特征。空间内容始终来自主干，不拷贝参考帧纹理。修正公式为 `F + alpha * gate * (contrast * (F - mean(F)) + shift * std(F))`，其中 `contrast` 和 `shift` 均在 [-1, 1]，最后投影零初始化。无效参考的修正为零，`residual_scale: 0` 关闭注入。

**新模式应启动一次新训练，不要直接用 `--resume` 加载旧 `latent_stats` 参考分支权重**：融合层形状不同，即使 `strict: false` 也不能跳过同名层的形状冲突。旧模式仍可通过原配置加载旧权重。

建议至少报告以下消融（固定数据、主干初始化、训练步数与评估流程，最终复核多个种子）：

| 实验 | `enabled` | `fusion_mode` | 说明 |
|---|---:|---|---|
| EvLight baseline | false | — | 完全不构建参考分支 |
| Reference affine | true | `reference_affine` | 有界亮度/对比度修正，当前训练配置 |
| Latent concat | true | `latent_stats` | 原参考均值调制 + `1x1` Linear |
| Spatial align | true | `spatial_align` | 旧 flow 对齐融合，仅用于消融 |

分别将 `loss.ssim_weight` 设为 0 和 0.1，并比较零注入与参考注入，可以区分结构约束和参考分支的收益；同一参考模型还应比较正确参考和打乱参考。请为不同实验指定不同输出目录。

以下参数分组保留供旧双层训练/微调流程使用，默认联合训练不分别更新它们：

- `model.meta_parameters()` / `model.reference_parameters()`：仅 `ReferenceBranch.frame_encoder`，即参考特征提取元表示 `omega`。
- `model.recovery_parameters()` / `model.task_parameters()`：EvLight、当前帧编码、统计门控和线性融合，即任务变量 `theta`。

双层训练的 outer loop 只更新参考帧特征提取器；任务微调冻结该编码器，重置并更新其余任务模块。

## 微调与推理

```bash
python finetune_event_dunet.py \
  --config configs/event_mini_dunet_finetune.yaml \
  --checkpoint outputs/event_mini_dunet_global/best.pt

python infer_event_dunet.py \
  --config configs/event_mini_dunet_infer.yaml \
  --checkpoint outputs/event_mini_dunet_affine_ssim/best.pt
```

微调示例沿用旧 `latent_stats` 配置及其旧权重，是可选的独立流程，不是当前联合训练的必需步骤。新模式直接使用训练与推理命令即可。恢复参考模型时必须保持 `fusion_mode`、`residual_scale`、SNR 设置一致，并使用严格加载核对参数。

更完整的设计依据和实验建议见 [docs/reference_branch.md](docs/reference_branch.md)。
"# adapt_second" 
