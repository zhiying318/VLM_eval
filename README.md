# VLM Evaluation Project

本项目用于评估 **InternVL3.5** 和 **Qwen2.5-VL** 等视觉-语言模型在简单数据集上的表现。

## 项目结构

```
VLM_Eval/
├── models/              # 模型加载模块
│   ├── __init__.py
│   └── model_loader.py  # ModelLoader 类
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── image_utils.py   # 图像预处理（从 SPARK 改编）
│   └── data_loader.py   # 数据加载器
├── data/                # 数据集目录
├── results/             # 评估结果
├── configs/             # 配置文件
├── evaluate.py          # 主评估脚本
├── requirements.txt     # Python 依赖
└── README.md           # 项目说明
```

## 从 SPARK 项目复用的代码

本项目从 SPARK benchmark 项目中提取并改编了以下有用代码：

1. **图像预处理** (`utils/image_utils.py`)
   - `dynamic_preprocess()`: 动态图像预处理，支持多分辨率
   - `load_image()`: InternVL 系列的图像加载函数
   - `build_transform()`: 图像变换管道

2. **模型加载** (`models/model_loader.py`)
   - InternVL3.5-8B 模型加载逻辑（从 SPARK test.py 改编）
   - 支持自动设备分配和内存优化

## 安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 准备数据集

### 创建示例数据集

```python
from utils import create_sample_dataset

# 创建示例数据集文件
create_sample_dataset("data/sample_dataset.json", num_samples=10)
```

### 数据格式

数据集支持 JSON 或 CSV 格式，需包含以下字段：

```json
[
  {
    "image_path": "path/to/image.jpg",
    "question": "What is in this image?",
    "answer": "A cat"
  }
]
```

## 使用方法

### 评估 InternVL3.5

```bash
python evaluate.py \
  --model internvl3.5 \
  --data_path data/your_dataset.json \
  --image_root data/images/ \
  --batch_size 1 \
  --results_dir results/
```

### 评估 Qwen2.5-VL

```bash
python evaluate.py \
  --model qwen2.5 \
  --data_path data/your_dataset.json \
  --image_root data/images/ \
  --batch_size 1 \
  --results_dir results/
```

### 使用自定义模型路径

```bash
python evaluate.py \
  --model internvl3.5 \
  --model_path /path/to/your/model \
  --data_path data/your_dataset.json \
  --batch_size 1
```

## 参数说明

- `--model`: 要评估的模型 (`internvl3.5` 或 `qwen2.5`)
- `--data_path`: 数据集文件路径（JSON 或 CSV）
- `--image_root`: 图像根目录（如果数据集中是相对路径）
- `--model_path`: 自定义模型路径（可选，默认使用官方模型）
- `--batch_size`: 批大小（默认为 1）
- `--results_dir`: 结果保存目录（默认为 `results/`）

## 输出

评估结果将保存为 JSON 文件，包含：

```json
{
  "metrics": {
    "total": 100,
    "correct": 85,
    "errors": 2,
    "accuracy": 0.87,
    "error_rate": 0.02
  },
  "results": [
    {
      "image_path": "...",
      "question": "...",
      "ground_truth": "...",
      "prediction": "...",
      "metadata": {}
    }
  ]
}
```

## 推荐的简单数据集

以下是一些适合测试的公开 VQA 数据集：

1. **VQAv2** - 经典 VQA 数据集
2. **GQA** - 场景图问答数据集
3. **COCO-VQA** - 基于 COCO 图像的问答
4. **OKVQA** - 需要外部知识的 VQA
5. **TextVQA** - 包含文字识别的问答

## 系统要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- GPU: 至少 16GB VRAM（用于 7B-8B 模型）
- RAM: 32GB+ 推荐

## 参考

- SPARK 项目: https://github.com/top-yun/SPARK
- InternVL3.5: https://github.com/OpenGVLab/InternVL (使用 InternVL3.5-8B 模型)
- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL

## License

本项目基于 SPARK 项目改编，遵循相应的开源协议。
