# 需要环境：pip install torch transformers

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image  # 如果是图像相似度计算需要PIL

# 加载预训练模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 示例文本（可替换为图像路径）
text1 = "00000000000000000000000000000000000000000000000000000000000"
text2 = "1111111111111111111111111111111111111"

# text1 = "Man with curly hair speaks directly to camera"
# text2 = "Kids exercise in front of parked cars"

# 编码文本特征
with torch.no_grad():
    inputs = processor(text=[text1, text2], return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)

# 计算余弦相似度
similarity = torch.nn.functional.cosine_similarity(text_features[0], text_features[1], dim=0).item()

print(f"相似度: {similarity:.2f}")

# 输出示例：相似度: 0.23（根据实际输入会有变化）