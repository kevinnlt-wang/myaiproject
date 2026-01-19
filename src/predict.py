# 模型推理脚本
# -提供多个方法，实现训练好的模型的运行测试，推理数据得到结果
#【注意】：
# 老师下发的 best_model.pt，是通过GPU训练的
# 默认情况你电脑用CPU是不能运行的，需要追加设置设备的代码
#------------------------------------------------------------

import torch
import config
from model import ReviewAnalyzerModel
from transformers import AutoTokenizer

# 批量预测
def predict_bath(model, inputs):
    # 开启预测
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    batch_result = torch.sigmoid(output) #将最终输出内容转换为0-1之间
    return batch_result.tolist()         #将整个结果作为列表返回


# 模型预测
def predict(user_input, model, tokenizer, device):
    #处理输入的内容
    inputs = tokenizer(
        user_input, 
        truncation=True, 
        padding=True, 
        max_length=config.SEQ_LEN,
        return_tensors='pt'
    )
    # 提取键值对内容
    inputs = {k:v.to(device) for k,v in inputs.items()}
    # 调用方法，执行批量预测，得到最终结果
    batch_result = predict_bath(model, inputs)
    # 返回结果的值，它是个列表
    return batch_result[0]


# 正式开始推理
def run_predict():
    # 1、确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2、加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    # 3、构建模型
    model = ReviewAnalyzerModel().to(device)

    # 4、加载训练后的权重文件
    model.load_state_dict(torch.load(config.MODEL_DIR/'best_model.pt', map_location=torch.device('cpu')))
    print('模型加载完毕！')

    # 5、运行
    print('=========欢迎使用评论分析系统!（exit退出）=========')
    while True:
        # 获取用户输入
        user_input = input('>：')
        # 是否退出
        if user_input.lower() == 'exit':
            print('感谢使用，再见！')
            break
        # 是否空
        if user_input.strip() == '':
            print('请输入内容')
            continue
        #执行模型推理
        result = predict(user_input, model, tokenizer, device)
        #判断结果
        if result > 0.5:
            print(f'正面评价（可信度）：{result}')
        else:
            print(f'负面评价（可信度）：{1-result}')


# 可执行
if __name__ == '__main__':
    run_predict()

