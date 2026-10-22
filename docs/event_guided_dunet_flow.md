# Event-guided Mini-DUNet 数据流

```text
illumination = IllumiinationNet(low, ill_list)
light_up = low * illumination + low

image_feature = ImageExtractor(light_up)
event_feature = EventExtractor(event)
holistic_feature = Conv1x1(image_feature || event_feature)

if ReferenceBranch.enabled:
    current_latent = current_encoder(light_up)                 # theta
    reference_latent = frame_encoder(img_refer)                # omega/meta
    styled_latent = std(current_latent) * norm(current_latent) \
                    + mean(reference_latent)
    channel_gate = Gate(stats(current_latent, reference_latent))
    mixed_latent = current_latent + channel_gate * (
        styled_latent - current_latent
    )
    residual = Conv1x1(current_latent || mixed_latent)
    holistic_feature += 0.1 * channel_gate * residual

pred = Unet_ReFormer(
    holistic_feature, light_up, image_feature, snr_map, event_feature
)
```

默认参考分支只迁移参考潜特征的通道均值，空间排列和标准差来自当前帧，因此不要求两帧几何对齐。它是 `ImageEnhanceNet.forward` 的一个可选调用参数；关闭时，`holistic_feature` 直接进入原 EvLight ReFormer，不改变 backbone 内部各层的通道数和拓扑。

元学习参数分组：

- `meta_parameters()` / `reference_parameters()`：仅 `ReferenceBranch.frame_encoder`，作为外层参考特征元表示 `omega`。
- `task_parameters()` / `recovery_parameters()`：EvLight、`current_encoder`、统计门控和线性融合，作为内层任务参数 `theta`。
- `reference_adapter_parameters()`：完整参考 adapter，仅供检查，不作为 optimizer 参数组。
- `degradation_parameters()`：仅为兼容旧调用保留的别名，当前等价于 `meta_parameters()`，不再指照明估计分支。

训练关系：

```text
inner/support: freeze(reference frame_encoder omega), update(all other theta)
outer/query:   freeze(all theta), update(reference frame_encoder omega)
fine-tune:     freeze(reference frame_encoder omega), reset/update(all theta)
```
