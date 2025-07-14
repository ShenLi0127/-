# 电力数据预测项目

基于PyTorch的电力消耗预测系统，支持短期（90天）和长期（365天）预测。

## 项目结构

```
Power_app/
├── datasets/               # 存放数据集和处理数据集的代码
├── model/                  # 模型
│   ├── lstm.py             # LSTM模型定义
│   ├── transformer.py      # transformer模型定义
│   └── saved/              # 存放保存的模型
├── output/                 # 存放输出结果和图表
├── script/                 # 存放脚本
│   ├── run.sh              # 运行程序的脚本
├── environment.yml         # conda环境配置
├── requirements.txt        # pip依赖文件
```

## 快速开始

### 环境设置

使用Conda:

```bash
conda env create -f environment.yml
conda activate power
```

使用pip:

```bash
pip install -r requirements.txt
```

### 运行脚本

```bash
bash ./script/run.sh
```

### 模型训练与预测

```bash
# lstm
python exp.py datasets/daily_train.csv datasets/daily_test.csv --model lstm

# transformer
python exp.py datasets/daily_train.csv datasets/daily_test.csv --model transformer

# lstm+多尺度
python exp.py datasets/daily_train.csv datasets/daily_test.csv --model lstm  --multiscale
```
