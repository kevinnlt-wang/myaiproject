# 模型定义脚本
# -自定义模型结构，适配当前业务的需求，同时进行各种参数的配置
# -直接基于bert预训练模型
# -这个模型操作方法目前没有头部的，这里需要自己进行处理，添加一个
#-----------------------------------------------------------------

import torch
import config
from transformers import AutoModel

# 评析模型类，继承于 torch.nn.Module
class ReviewAnalyzerModel(torch.nn.Module):

    # self是当前类对象，是torch.nn.Module子类
    # self.linear, self.bert 是父类中存在的，名字不能错

    # 初始化
    def __init__(self):
        # 满足语义的要求，先执行父类初始化
        super(ReviewAnalyzerModel, self).__init__()
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        # 创建线性层，作为分类的头部，实现最终的输出
        # 注意：bert-base-chinese模型的隐藏层输出的维度是768
        # 隐藏的维度与当前self.bert.config.hidden_size要保持一致，对接模型
        # 1表示输出的维度，我们不要默认隐藏层的768个结果，只要1个结果，好的还是坏的就可以了
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 1)

    # 模型前向传播过程
    # -输入的句子经过bert模型，输出的特征向量的维度768
    # -经过自定义的线性层后，输出的维度1
    def forward(self, input_ids, attention_mask, token_type_ids):        
        # 输出的结果是一个实数，表示这个句子的情感倾向
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # 提取最后一个隐藏层的状态 = [batch_size, seq_len, hidden_size]
        last_hidden_state = output.last_hidden_state

        # 去除原本没有用的参数，中间的那个值不要了:seq_len
        cls_hidden_state = last_hidden_state[:, 0, :]

        # 提取结果
        output = self.linear(cls_hidden_state).squeeze(-1)
        # 返回
        return output

       



