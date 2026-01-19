# 下载大模型
from transformers import AutoModel,AutoTokenizer

# 模型名称和下载目录
name = 'google-bert/bert-base-chinese'
dir = './cache/bert-base-chinese'

# 加载模型
AutoModel.from_pretrained(name, cache_dir=dir)
# 加载分词器
AutoTokenizer.from_pretrained(name, cache_dir=dir)

# 将模型文件复制到自己的预训练模型目录下
# /pretrained/bert-base-chinese/下面