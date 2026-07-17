# PDF 论文与 C-MAPSS 代码对应关系

## 一、论文中真实的 MetaFluAD 结构

论文 Figure 1 和公式（2）—（9）表明，MetaFluAD 的核心由四部分组成：

1. SEBlock 对 ProtVec 序列特征进行通道重标定，输出 256 维节点特征。
2. GAT（图注意力网络）分支依据抗原差异图聚合邻居信息，对应公式（2）—（5）。
3. Multi-head self-attention（多头自注意力）分支与 GAT 并行，对应公式（6）。
4. 两个 256 维分支拼接成 512 维节点表示。两个节点表示再拼接成 1024 维，由 MLP 预测抗原距离，对应公式（7）。
5. Reptile 在不同流感亚型任务上执行任务级更新与全局参数更新，对应公式（8）、（9）。

## 二、迁移到 C-MAPSS 后的实现

| MetaFluAD | Meta-GNN-RUL 代码 | 说明 |
|---|---|---|
| 流感亚型 | FD001—FD004 | 每个子数据集是一个元任务 |
| 病毒节点 | 发动机时间窗口节点 | 一个窗口代表某发动机当前退化状态 |
| HA1 ProtVec | 多传感器滑动窗口 | 默认 50×14 |
| SEBlock | `models/se_block.py` | 学习传感器通道权重 |
| 抗原差异图 | 退化相似图 | 余弦近邻或 DTW 近邻 |
| GAT 分支 | `models/gat_encoder.py` | 聚合相似退化节点信息 |
| 自注意力分支 | `models/sensor_attention.py` | 在 LSTM 序列表示上执行多头自注意力 |
| 512维病毒表示 | 512维退化表示 | 两个 256维分支拼接 |
| 成对抗原距离 | RUL 主任务 + 成对退化距离辅助任务 | 主输出为 RUL；辅助目标为两个窗口 RUL 差的绝对值 |
| Reptile | `meta_learning/reptile.py` | 跨 FD 子数据集学习共享初始化 |

## 三、为什么不能完全照搬论文

原论文预测的是两个病毒之间的距离，而 C-MAPSS 的标准任务是预测单个发动机当前窗口的 RUL。因此代码把单窗口 RUL 回归设为主任务。为了保留论文的成对距离思想，代码通过 `pair_aux_weight` 加入可关闭的成对退化距离辅助损失，但它不会取代正式 RUL 指标。

论文的图由已有的抗原距离构造；测试时若直接用真实 RUL 构图会发生标签泄漏。因此本项目只使用模型 Embedding（嵌入表示）的余弦相似度或原始传感器轨迹 DTW 构图，绝不使用测试 RUL 建边。

## 四、论文公式在代码中的位置

- 公式（2）—（5）：`models/gat_encoder.py` 的 `DenseEdgeGATLayer`。
- 公式（6）：`models/sensor_attention.py` 的 `TemporalSelfAttention`。
- 公式（7）：`models/rul_predictor.py` 与 `pairwise_predictor`。
- 公式（8）：`meta_learning/reptile.py` 的 `inner_adapt`。
- 公式（9）：`meta_learning/reptile.py` 的 `reptile_meta_step`。

## 五、实验边界

本工程实现的是“受 MetaFluAD 启发、面向 C-MAPSS 的迁移方法”，不是 MetaFluAD 原论文在流感数据上的逐行复现。正式论文中应将其命名为 Meta-GNN-RUL，并明确哪些模块来自 MetaFluAD、哪些模块为面向 RUL 的适配设计。
