# Meta-GNN-RUL：C-MAPSS 跨工况少样本剩余寿命预测

本工程同时依据 PDF 论文《MetaFluAD: meta-learning for predicting antigenic distances among influenza viruses》和《Meta_GNN_RUL_CMAPSS完整代码实现方案》实现。主模型为 **Sensor SEBlock（传感器通道重标定）+ LSTM（长短期记忆网络）+ GAT（图注意力网络）/Multi-head Self-Attention（多头自注意力）并行分支 + Reptile（元学习）**，用于 C-MAPSS 的跨工况、少样本 RUL（Remaining Useful Life，剩余使用寿命）预测。

论文原方法与本工程的逐项对应、公式位置和必要改动见 `PAPER_TO_CODE_MAPPING.md`。

## 1. 已实现内容

- C-MAPSS FD001–FD004 数据读取、训练/测试 RUL 标签生成。
- 训练域统计量拟合的 Z-score/Min-Max 归一化，避免目标测试数据泄漏。
- 滑动窗口样本生成；测试集仅取每台发动机最后一个窗口，符合官方评估方式。
- 纯 PyTorch 实现的批内 kNN（k近邻）退化图与多头 GAT，无需安装 PyTorch Geometric；支持余弦相似图和 DTW（动态时间规整）图。
- 与论文 Figure 1 一致的 GAT/自注意力并行结构，两个 256 维分支拼接成 512 维退化表示。
- 可选成对退化距离辅助损失，用于保留论文“两个节点拼接后预测距离”的思想。
- Reptile 内循环任务适应与外循环元初始化更新。
- 目标域按发动机编号抽取少样本支持集，和测试集严格分离。
- 5 个对照方法：LSTM、CNN-LSTM、Transformer、GNN-RUL、Reptile-LSTM。
- 消融：去 GAT、去传感器注意力、去 Reptile。
- RMSE（均方根误差）、MAE（平均绝对误差）、R²（决定系数）、NASA Score（NASA非对称评分）。

## 2. 数据放置

请将 NASA C-MAPSS 解压为以下结构：

```text
data/
├── FD001/train_FD001.txt, test_FD001.txt, RUL_FD001.txt
├── FD002/train_FD002.txt, test_FD002.txt, RUL_FD002.txt
├── FD003/train_FD003.txt, test_FD003.txt, RUL_FD003.txt
└── FD004/train_FD004.txt, test_FD004.txt, RUL_FD004.txt
```

也支持把全部 txt 文件直接放在 `data/` 下。代码不会从网络自动下载数据，以便遵守数据集许可与保证实验可复现。

## 3. 环境与运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py --model meta_gnn
```

在没有真实数据时，可先做流程冒烟测试：

```bash
python scripts/generate_mock_data.py
python main.py --model meta_gnn
```

模拟数据只能检查代码是否能运行，不能用于报告科研结论。

## 4. 实验命令

```bash
# 所有 baseline（基线）与主模型，目标域 FD004
python main.py --suite baselines --target FD004

# 消融实验
python main.py --suite ablation --target FD004

# 按原论文的 5% / 10% / 20% 数据比例进行目标域适应
python main.py --model meta_gnn --target FD004 --support-ratio 0.05
python main.py --model meta_gnn --target FD004 --support-ratio 0.10
python main.py --model meta_gnn --target FD004 --support-ratio 0.20

# 四个目标域依次做 leave-one-domain-out（留一域验证）
python run_all_domains.py

# 多随机种子示例
python main.py --suite baselines --target FD004 --seed 0
python main.py --suite baselines --target FD004 --seed 1
python main.py --suite baselines --target FD004 --seed 2

# 一次运行 5 个随机种子并汇总均值±标准差
python scripts/run_multiseed.py --target FD004 --seeds 0 1 2 3 4
```

模型权重、数据划分中的发动机编号及指标保存在 `outputs/`。关键参数在 `configs/default.yaml` 中修改。

安装开发依赖后可执行测试：

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 5. 实验逻辑

默认设置把 FD001、FD002、FD003 作为源任务，FD004 作为目标任务。Reptile 在三个源任务中学习共享初始化；随后从 FD004 训练集中抽取总计 5 台有标签发动机，其中一小部分用于验证、其余用于少样本适应；最终在 FD004 官方测试集上评估。指定 `--target FD001` 时，源任务会自动切换为其余三个子集。

普通基线只使用相同的目标域支持集训练，因此能够与元学习模型公平比较。`gnn` 是去除 Reptile 的 GNN-RUL，也是“去元学习”消融。

## 6. 重要实现说明

- 图节点是当前 mini-batch（小批次）中的时间窗口表示，而不是整台发动机的固定节点。这样可以端到端训练且控制显存；每个节点由 LSTM 编码，一个节点的边连接到表示最相似的 k 个节点。
- 图的近邻选择基于 `detach()` 后的表示，边选择本身不可导，但 GAT 消息传递和预测损失可正常反向传播。
- DTW 图使用下采样后的多传感器均值退化轨迹构图，速度明显慢于余弦图，因此默认使用余弦相似度；`--suite ablation` 会同时运行二者。
- `pair_aux_weight` 控制论文成对距离思想对应的辅助损失；设为 0 即完全关闭。
- 源域合并统计量用于归一化，目标测试集从不参与归一化器拟合。
- C-MAPSS 常用的恒定/近恒定传感器已从默认 21 个传感器中剔除；可在 YAML 中恢复或修改。

## 7. 推荐正式实验

至少运行 4 个目标域 × 5 个随机种子，报告均值±标准差；同时记录每次目标支持发动机编号。建议进一步比较 1、3、5、10 台目标域支持发动机，并对图的 k 值做敏感性分析。论文写作时不要把 mock（模拟）数据结果当作真实实验结果。
