from transformers import T5ForConditionalGeneration, T5Config, RobertaTokenizer, PretrainedConfig, PreTrainedModel, \
    PreTrainedTokenizer
import torch.nn as nn
from typing import Optional, Tuple
import logging
import numpy as np
import torch
import os

logger = logging.getLogger(__name__)

MODEL_CLASSES = (T5Config, T5ForConditionalGeneration, RobertaTokenizer)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args) -> Tuple[PretrainedConfig, PreTrainedModel, PreTrainedTokenizer]:
    config_class, model_class, tokenizer_class = MODEL_CLASSES
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # load embedding_model
    embedding_model = EmbeddingModel.from_pretrained(args.model_name_or_path)
    embedding_model._set(tokenizer)
    if args.checkpoint_model_name is not None:
        model_path = os.path.join(args.output_dir, "{}".format(args.checkpoint_model_name))
        embedding_model.load_state_dict(torch.load(model_path))
        logger.info(f"*** load model checkpoint from {model_path} ***")
    embedding_model.to(args.device)

    logger.info("Finish loading embedding_model [%s] from %s", get_model_size(embedding_model), args.model_name_or_path)

    return config, embedding_model, tokenizer


class EmbeddingModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def _set(self, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids: Optional[torch.LongTensor] = None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
        return nn.functional.normalize(hidden_state, p=2, dim=1)
