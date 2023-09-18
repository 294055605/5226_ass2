from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch


class IndexingTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, **kwds):# 此方法初始化类并调用父类的初始化方法。它还设置了一个新的属性restrict_decode_vocab，这是一个函数，用于在解码时限制模型的输出。
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
    # 此方法计算模型的损失。它接收模型、输入数据和一个return_outputs参数。如果return_outputs为True，它将返回损失和一个假的输出列表；否则，它只返回损失。
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss
    # 此方法进行预测步骤。它接收模型、输入数据、一个prediction_loss_only参数和ignore_keys参数。
    # 在此方法中，模型被设置为评估模式，然后进行预测。使用model.generate方法进行预测，并使用提前停止和restrict_decode_vocab函数来限制输出
    # 最后，返回一个包含None、预测的文档ID和真实标签的元组。
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        with torch.no_grad():
            # greedy search
            doc_ids = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                early_stopping=True,)
        return (None, doc_ids, inputs['labels'])

