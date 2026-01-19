# 配置文件
from pathlib import Path

# 目录配置
ROOT_DIR = Path(__file__).parent.parent     #当前项目根路径
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'    #原始数据目录
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed' #预处理后的数据目录
MODEL_DIR = ROOT_DIR / 'model'              #训练好的模型目录
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'   #预训练模型目录

# 模型参数配置
SEQ_LEN = 128          #序列长度：输入模型的每个样本，最多包含128个tokens
BATCH_SIZE = 8         #批量大小：每次训练时，一次进入模型的样本的数量
EMBEDDING_DIM = 128    #词嵌维度：每个token被隐射为一个128维的向量
HIDDEN_SIZE = 256      #隐藏层大小：模型内部的神经网络是层的，最后一层(隐藏层)提供结果的，结果单元256个
LEARNING_RATE = 1e-5   #学习率：优化器，优化每一步更新参数的步长 0.00001
EPOCHS = 10            #训练轮数：整个训练数据集的操作将执行 10 次

