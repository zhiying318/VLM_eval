# test_internvl.py
from models import ModelLoader

# 测试模型加载
loader = ModelLoader("internvl3.5", device="cuda") # or qwen2.5
model, tokenizer = loader.load_model()

print("InternVL3.5 模型加载成功！")
print(f"模型设备: {next(model.parameters()).device}")

