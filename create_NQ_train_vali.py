import datasets
from datasets import load_from_disk, concatenate_datasets
import random
import numpy as np
import json
import os
import pyarrow as pa
import shutil
'''
这段代码是一个Python脚本，它涉及到处理Natural Questions (NQ)数据集的过程.
总的来说，这个脚本的目的是从Natural Questions数据集中随机选择文档，然后将这些文档和相关的问题分为两个文件：一个用于多任务训练，另一个用于验证。
'''
# 设置随机种子和数据集大小
random.seed(313)

NUM_TRAIN = 6000
NUM_EVAL = 1830
# 这里设置了随机种子以确保随机过程的可重复性，并定义了训练数据和验证数据的大小。
# 加载Natural Questions数据集
# 这行代码使用datasets库加载Natural Questions数据集的训练部分。
# data = datasets.load_dataset('natural_questions',
#                              token="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",
#                              cache_dir="E:/pycharm_gc/pytorch/task/DSI-transformers-main/datasets",
#                              split='train[:100000]')  # 这里仅下载训练数据的前10%。

# 获取训练和验证数据集的目录路径
train_path = "E:/pycharm_gc/pytorch/task/DSI-transformers-main/datasets/train"
valid_path = "E:/pycharm_gc/pytorch/task/DSI-transformers-main/datasets/valid"

data = load_from_disk(dataset_path=valid_path)
print('dataset finished')


# 随机洗牌数据集索引，这里首先为整个数据集创建一个索引列表，然后随机洗牌这些索引。
rand_inds = list(range(len(data)))
random.shuffle(rand_inds)
# 初始化标题集合和文档ID
title_set = set()# title_set：用于存储已经处理过的标题，以确保每个标题只处理一次。
current_docid = 0# current_docid：用于为每个文档分配一个唯一的ID。

# 打开输出文件并处理数据
# 这里使用Python的with语句同时打开两个文件：一个用于训练数据(tf)，另一个用于验证数据(vf)。
with open('NQ_10k_multi_task_train.json', 'w') as tf, \
        open('NQ_10k_valid.json', 'w') as vf:
    print('starting...')
    for ind in rand_inds:# 遍历随机化的索引
        # 提取每个文档的标题，并检查该标题是否已在title_set中
        print(f'start {ind}')
        title = data[ind]['document']['title']  # we use title as the doc identifier to prevent two docs have the same text
        if title not in title_set:
            title_set.add(title)# 处理新文档并添加标题到title_set, 确保之后不会重复处理

            # 提取文档的tokens并组合成文本
            token_inds = np.where(np.array(data[ind]['document']['tokens']['is_html']) == False)[0]# 使用np.where找到非HTML tokens的索引, 是个元组
            tokens = np.array(data[ind]['document']['tokens']['token'])[token_inds]# 使用这些索引从tokens中提取出非HTML的文本内容。
            doc_text = " ".join(tokens)# 使用join方法将这些tokens组合成一个完整的文档文本。
            question_text = data[ind]['question']['text']# 提取问题文本,从当前数据中提取与文档相关的问题文本。
            # 将文档和相关问题写入相应的输出文件:将文档文本和问题文本分别转换为JSON格式，并为每个文档/问题分配一个唯一的ID。

            # 然后，基于title_set的长度（即已处理的文档数量），决定将数据写入训练文件还是验证文件。
            jitem = json.dumps({'text_id': str(current_docid), 'text': 'document: ' + doc_text})
            tf.write(jitem + '\n')
            jitem = json.dumps({'text_id': str(current_docid), 'text': 'question: ' + question_text})
            if len(title_set) <= NUM_TRAIN:
                tf.write(jitem + '\n')
            else:
                vf.write(jitem + '\n')
            current_docid += 1# 更新文档ID, 为下一个文档增加文档ID
            # 检查是否已经处理了足够数量的文档
            if len(title_set) == NUM_TRAIN + NUM_EVAL:# 如果已处理的文档数量达到了预定的训练和验证数据的总量，则停止循环。
                break
        print(f"Creating training and validation dataset: {'{:.1%}'.format(len(title_set)/(NUM_TRAIN + NUM_EVAL))}", end='\r')