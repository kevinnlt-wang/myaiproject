# 模型训练脚本
# -正式开始训练模型
#---------------------------------------------------------------
import torch
import config
from dataset import get_dataloader
from model import ReviewAnalyzerModel
from tqdm import tqdm


# 模型微调
# -训练的每一轮模型样本在这里进行整体处理
# -主要的目的是用来更新模型的权重
def train_one_epoch(model, dataloader, loss_fun, optimizer, device):
    '''训练一个epoch的逻辑
    Args:
        model：模型
        dataloader：数据加载器(训练数据)
        loss_fun：损失函数
        optimizer：优化器
        device：设备
    Return:
        当前epoch的平均损失
    '''
    
    # 开启模型训练
    model.train()

    # 默认的总的损失值：0（0没有损失，完全符合）
    total_loss = 0

    # 循环遍历
    for batch in tqdm(dataloader, desc='训练中...'):
        # 取出每一个样本数据的键值，key是特征，value是目标
        inputs = {k:v.to(device) for k,v in batch.items()}
        # 提取里面的目标值
        labels = inputs.pop('labels').to(dtype=torch.float, device=device)
        
        # 前向传播
        # -提供数据给模型，通过神经网络学习，得到一个结果
        outputs = model(**inputs)

        # 计算损失值
        # -outputs的结果与真实结果的差值就是损失值
        loss = loss_fun(outputs, labels)

        # 反向传播
        # -反向回来，一层一层的返回，诸葛更新当前的情况，保留下来
        loss.backward()             #开始反向
        optimizer.step()            #优化器-默认步长
        optimizer.zero_grad()       #优化器-梯度清0
        total_loss += loss.item()   #累加每一轮的损失值

    # 返回总的损失值比例
    return total_loss / len(dataloader)



# 正式训练
def train():
    # 1、设备：GPU(N卡，安装cuda)还是CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2、数据集
    dataloader = get_dataloader(train=True)

    # 3、构建模型
    model = ReviewAnalyzerModel().to(device)

    # 4、损失函数
    loss_fun = torch.nn.BCEWithLogitsLoss()

    # 5、优化器(微调)
    # -我们使用的是MertModel，它设置的权重参数会很多很多
    # -我们这里调整一下，把学习的效率设置的底一点，一般 1e-5 == 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6、判断情况
    # -目前不知道最终合格的损失率是多少，就定义一个无穷大的浮点小数
    # -至于什么时候可以满足，后面自己定义
    best_loss = float('inf')

    # 7、循环训练：10轮
    for epoch in range(1, config.EPOCHS+1):
        print(f'=============[Epoch{epoch}]=============')
        # 调用方法
        train_loss = train_one_epoch(model, dataloader, loss_fun, optimizer, device)
        print(f'train_loss：{train_loss:.4f}')
        # 判断是否满足要求
        if train_loss < best_loss:
            #更新损失率
            best_loss = train_loss
            # 保存好当前的训练好的模型
            torch.save(model.state_dict(), config.MODEL_DIR / 'best_model.pt')
            print('最佳的模型以生成！')


# 可执行
if __name__ == '__main__':
    train()