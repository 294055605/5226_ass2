from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm


class QueryEvalCallback(TrainerCallback):# 定义一个新的回调类，用于在训练中的某些时刻执行特定操作。
    # 初始化方法需要五个参数：测试数据集、日志记录器、解码词汇表的限制、训练参数和分词器。
    # __init__ 方法初始化这个回调类，并为后续操作准备数据加载器
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        # 这些行将传入的参数保存为类的属性。
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )
    # 这部分代码创建了一个数据加载器，用于加载测试数据集。定义了一个在每个训练周期结束时调用的方法。
    # on_epoch_end 方法在每个训练周期结束时被调用，用于评估模型的性能。
    def on_epoch_end(self, args, state, control, **kwargs):
        # 初始化两个变量用于评估模型性能。
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()# 从传入的关键字参数中获取模型并将其设置为评估模式
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset)})

# 此函数的目的是计算模型的准确性。它接收一个参数 eval_preds，其中包含模型的预测值 (eval_preds.predictions) 和真实标签 (eval_preds.label_ids)。
# 函数的逻辑是遍历每一个预测值和对应的真实标签，并判断预测是否正确。最终返回模型的准确率。
def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}

# main 函数是这段代码的主要执行部分，它负责模型的初始化、训练和评估。
def main():
    # 设置模型名和文档长度,这里指定了要使用的预训练模型名称（T5-large）和文档的最大长度（32 tokens）。
    model_name = "t5-small"
    L = 32  # only use the first 32 tokens of documents (including title)

    # 初始化Weights & Biases
    # 使用 Weights & Biases（通常简称为 wandb）进行实验跟踪和日志记录。这两行代码登录 wandb 账户并初始化一个新的实验。
    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project="DSI", name='NQ-10k-t5-small')
    # 加载模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='E:\APP\study\pytorch\env\model')
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='E:\APP\study\pytorch\env\model')
    # 加载训练、评估和测试数据集。这里注意，eval_dataset 实际上是用于报告模型是否可以记忆（索引）所有训练数据点的，而真正的评估集是 test_dataset。
    train_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_multi_task_train.json',
                                         max_length=L,
                                         cache_dir=r'E:\pycharm_gc\pytorch\task\DSI-transformers-main\data\NQ\cache',
                                         tokenizer=tokenizer)
    
    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_multi_task_train.json',
                                        max_length=L,
                                        cache_dir=r'E:\pycharm_gc\pytorch\task\DSI-transformers-main\data\NQ\cache',
                                        tokenizer=tokenizer)
    
    # This is the actual eval set.
    test_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_valid.json',
                                        max_length=L,
                                        cache_dir=r'E:\pycharm_gc\pytorch\task\DSI-transformers-main\data\NQ\cache',
                                        tokenizer=tokenizer)

    ################################################################
    # 这部分代码创建了一个列表，其中包含允许模型生成的token IDs。目标是确保模型只生成整数文档ID。
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []

    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################
    # 设置训练参数,定义了训练的参数，例如学习率、批次大小、评估策略等。
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        evaluation_strategy='steps',
        eval_steps=1000,
        max_steps=1000000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=50,
        save_strategy='steps',
        save_steps=500,
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=10,
        # gradient_accumulation_steps=2
    )
    # 初始化和启动训练器, 首先，使用给定的参数、模型、数据集等初始化一个自定义训练器。然后，调用 train() 方法开始训练过程。
    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab
    )
    trainer.train()


if __name__ == "__main__":
    main()
