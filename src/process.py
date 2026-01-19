# 数据预处理脚本
# -负责将原始数据csv进行预处理，生成模型可以使用的：train,test
# ------------------------------------------------------------
import config
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

# 预处理方法
def process():
    print('----------> 开始数据处理...')

    # 1、读取文件
    # -读取后是一个对象，默认是作为train训练集，我们要提取这个训练集
    dataset = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']

    # 2、过滤数据
    # -删除无关字段
    dataset = dataset.remove_columns(['cat'])
    # -过滤异常数据
    dataset = dataset.filter(lambda row: row['review'] is not None and row['review'].strip() != '' and row['label'] in [0, 1])

    # 3、划分数据集
    # -label列原始数据0和1，将来要作为2个分类，就是0和1两个类别
    # -转换label这个列的数据，变成目标值类型
    dataset = dataset.cast_column('label', ClassLabel(num_classes=2))
    # -从训练集train中，切分20%出来作为test测试集
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')

    # 4、分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    # 5、内部函数：构建所有数据集-只有当前函数自己可以调用
    def batch_tokenize(batch):
        inputs = tokenizer(
            batch['review'],
            max_length=config.SEQ_LEN,
            truncation=True,
            padding='max_length'
        )
        inputs['labels'] = batch['label']
        return inputs
    # -执行map构建，得到结果
    dataset_dict = dataset_dict.map(batch_tokenize, batched=True, remove_columns=['review','label'])
    
    # 6、保存数据集
    dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)

    print('----------> 数据处理-完成！')


# 是否主程序运行(导入时不会执行)
if __name__ == '__main__':
    process()