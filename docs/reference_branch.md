# 参考帧特征分支

四个数据集配置均使用 `reference_affine` 模式。此次整理保留 `ReferenceFeatureInjector` 的网络构造、初始化及前向计算，仅清理原双层训练使用的参数分组和任务重置辅助接口。

## 特征计算

1. `frame_encoder` 编码正常光参考 RGB，`current_encoder` 编码当前帧的 light-up RGB。二者结构相同，构造时复制初始权重，之后作为独立参数共同训练。
2. 分别计算当前和参考特征的通道均值、标准差，拼接后输入门控和仿射参数预测模块。
3. 门控使用 `sigmoid`，预测的对比度与偏移参数使用 `tanh` 限幅。
4. 对主干的 holistic feature 做一次残差修正，然后进入原 `Unet_ReFormer`。空间内容来自主干，参考帧通过通道统计影响修正。

令主干特征为 `F`，其通道统计为 `mean(F)` 和 `std(F)`，修正为：

```text
gate = sigmoid(gate_network(current_stats || reference_stats)) × reference_valid
contrast, shift = tanh(affine_network(current_stats || reference_stats))
output = F + residual_scale × gate × (
    contrast × (F - mean(F)) + shift × std(F)
)
```

均值和标准差按每个样本、每个通道在空间维度计算。主干修正使用 FP32 计算，再转回输入特征类型；这保持了现有混合精度行为。

## 初始化与有效性

当前配置如下：

```yaml
ReferenceBranch:
  enabled: true
  encoder_blocks: 2
  fusion_mode: reference_affine
  residual_scale: 0.1
  statistics_eps: 0.000001
  zero_init: true
```

`fusion_out` 零初始化，因此起始参考修正为零。第一步先更新最后投影，后续步骤再向上游编码器传播非零梯度。`reference_valid: false` 会将该样本门控置零；`residual_scale: 0` 也可关闭注入。`enabled: false` 则完全不构建参考分支。

`latent_stats` 和 `spatial_align` 的已有实现继续保留，供匹配历史权重或进行结构消融；当前四个配置明确选择 `reference_affine`。不同模式的融合层形状可能不同，加载历史权重前必须匹配 `fusion_mode`，非严格加载也不会忽略同名参数的尺寸不一致。

## 联合训练

EvLight 照明网络、图像–事件恢复网络、当前帧编码器、参考帧编码器、门控和融合层全部交给同一个 AdamW 优化器。每批数据只计算一个重建目标，并共同更新全部参数。

当前重建损失为 `Charbonnier + 0.5 × LPIPS`。Charbonnier 的实现保持 `sqrt(mean_channel((pred - gt)²) + eps).mean()`，`eps: 0.0001`；LPIPS 使用 AlexNet、`normalize=True`，其特征网络权重冻结。可通过 `loss.ssim_weight` 开启原有 SSIM 损失，四份默认配置均设为 `0`。

网络数据流见 [event_guided_dunet_flow.md](event_guided_dunet_flow.md)，数据集配置与运行方式见 [README](../README.md)。
