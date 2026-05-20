# 时域插值分支与频域分支融合方案

## Summary

在现有频域输入 `[N, 3, 29, 128]` 基础上，新增一个原始时域分支。时域分支按傅里叶变换相同的 29 个滑动窗口切分原始 3000 点 epoch，并将每个 200 点窗口插值重采样为 128 点，得到 `[N, 3, 29, 128]`。模型内对同一通道、同一时间窗的频域和时域 token 做 `concat + Linear` 融合，再进入现有 Transformer 主干。

## Data Shape

现有频域分支：

```text
raw epoch [3000]
-> STFT with 2s window / 1s step
-> [29, 128]
```

新增时域分支：

```text
raw epoch [3000]
-> same 29 sliding windows
-> each window [200]
-> interpolation resample to [128]
-> [29, 128]
```

三通道加载后：

```text
freq: [N, 3, 29, 128]
time: [N, 3, 29, 128]
```

推荐进入模型的数据结构：

```text
x: [N, 3, 2, 29, 128]
```

其中：

```text
3 = EEG_Fpz-Cz / EEG_Pz-Oz / EOG
2 = freq / time
29 = 时间窗数量
128 = 每个时间窗的特征维度
```

## Fusion Rule

第一版采用简单、可审计、最小改动的 `concat + Linear projection`。

对每个通道、每个时间窗单独融合：

```text
freq_token: [128]
time_token: [128]

concat(freq_token, time_token) -> [256]
Linear(256, 128) -> fused_token [128]
```

批量形式：

```text
freq_channel: [B, 29, 128]
time_channel: [B, 29, 128]

concat on feature dim -> [B, 29, 256]
Linear(256,128) -> [B, 29, 128]
```

融合后模型仍然得到三个通道：

```text
x1: [B, 29, 128]
x2: [B, 29, 128]
x3: [B, 29, 128]
```

因此后续 positional encoding、单通道 Transformer、cross-attention、多通道 Transformer 和分类头可以保持原有结构。

## Why This Fusion

- 不直接拼接后进入 Transformer，因为原模型期望 `dim_model=128`，裸拼接会变成 256 维，牵连修改范围较大。
- `Linear(256,128)` 可以让模型学习频域和时域的组合方式，同时保持原 Transformer 输入维度不变。
- 融合发生在同一通道、同一时间窗内部，语义最清晰：第 `t` 个频域 token 只和第 `t` 个时域 token 融合。
- 相比双 Transformer 后 late fusion，这个版本参数更少、改动更小，更适合作为第一版实验 baseline。

## Audit Points

- 时域窗口必须和 STFT 窗口完全一致：`window=200`、`step=100`、`num_windows=29`。
- 时域分支不是 `3000 -> 29`，而是 `3000 -> 29 x 128`。
- 插值重采样只做长度变换，不应改变标签、不应打乱 epoch 顺序。
- 频域和时域应分别归一化，避免两种模态数值分布互相污染。
- `data_loader` 必须保证 `freq`、`time`、`labels` 的样本顺序完全一致。
- 模型融合后每个通道仍必须是 `[B, 29, 128]`。
- 第一版只做 `concat + Linear`，暂不引入门控融合、cross-modal attention 或双主干，以便控制变量。
- 必须保留仅频域 baseline，后续用同一训练配置比较频域模型与频域+时域模型。

## Assumptions

- 原始 epoch 长度仍为 30 秒、采样率 100Hz、共 3000 点。
- 当前频域输出 `[29,128]` 继续保留。
- 当前 Transformer 配置中的 `pad_size=29`、`dim_model=128` 保持不变。
- 时域分支用于补充原始波形形态信息，不替代傅里叶分支。
