# 自定义dataset
# -读取处理后的数据集，封装成模型可以识别的对象，提供预训练模型使用
#----------------------------------------------------------------

import config
from datasets import load_from_disk
from torch.utils.data import DataLoader

# 获取数据集对象
def get_dataloader(train=True):
    path = str(config.PROCESSED_DATA_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(path)
    dataset.set_format(type='torch') #设置数据格式为torch张量类型
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


# 测试一下看看
if __name__ == '__main__':
    print('===>预览形状：')
    train = get_dataloader(train=True)
    test = get_dataloader(train=False)
    print(len(train))
    print(len(test))

    for batch in train:
        for k,v in batch.items():
            print(k, '-->', v.shape)
        break