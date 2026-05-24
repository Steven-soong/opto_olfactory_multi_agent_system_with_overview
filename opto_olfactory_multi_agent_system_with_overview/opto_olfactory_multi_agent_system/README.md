# Optoelectronic Fusion Olfactory Data Analysis System

这是一个“基于多智能体协同的光电融合智能嗅觉数据分析与预警系统”的基础可运行原型。

## 系统功能

- 模拟电子鼻高频时序电信号
- 模拟光谱图像/光学传感数据
- 使用双分支 CNN 提取跨模态特征
- 使用多 Agent 协作完成：
  - 特征提取与降噪 Agent
  - 长链推理与融合决策 Agent
  - 安全合规预警 Agent
- 输出结构化安全预警报告 JSON


## 项目说明补充

本项目已经在 `docs/project_overview.md` 中补充以下两个核心要素：

1. **项目解决的核心痛点**：包括跨模态数据割裂、高维信号噪声、复杂混合气体误判、实时安全预警不足、过度依赖人工经验等问题。
2. **核心逻辑流**：系统包含多 Agent 协作架构和工程化长链推理流程，具体包括特征提取 Agent、长链推理与融合决策 Agent、安全合规预警 Agent。

核心逻辑流如下：

```text
电子鼻时序信号 + 光谱图像数据
        ↓
特征提取 Agent
        ↓
跨模态特征向量 + 初步分类概率
        ↓
长链推理与融合决策 Agent
        ↓
贝叶斯后验概率 + 化学知识图谱推理结果
        ↓
安全合规预警 Agent
        ↓
结构化预警报告 + 处置建议
```


## 项目结构

```text
opto_olfactory_multi_agent_system/
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_simulator.py
│  ├─ models.py
│  ├─ agents.py
│  ├─ train.py
│  └─ run_demo.py
├─ checkpoints/
└─ outputs/
```

## 安装环境

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

如果你使用 Anaconda：

```bash
conda create -n olfactory_ai python=3.10
conda activate olfactory_ai
pip install -r requirements.txt
```

## 训练基础模型

```bash
python -m src.train
```

训练完成后会生成：

```text
checkpoints/fusion_model.pt
```

## 运行多 Agent 预警演示

```bash
python -m src.run_demo
```

运行后会生成：

```text
outputs/demo_report.json
```

## 重要说明

本项目是教学/科研原型，不可直接作为真实实验室安全报警系统使用。真实部署必须替换为：
1. 实测传感器数据；
2. 真实光谱库和电子鼻响应库；
3. 标准化标定曲线；
4. 经实验室审定的安全规范与危废处置流程；
5. 经过验证的报警阈值、误报率和漏报率评估。
