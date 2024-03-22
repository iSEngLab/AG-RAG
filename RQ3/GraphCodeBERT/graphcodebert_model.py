import torch.nn as nn
import torch
import numpy as np
import os
from transformers import (RobertaConfig, RobertaTokenizer, RobertaModel)
from typing import Optional
import logging


class DataBase(object):
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.history = []

    def __len__(self):
        return self.vector.shape[0]

    def __getitem__(self, index):
        return self.vector[index]

    def search(self, query, number, stage=None):
        scores = np.matmul(query, self.vector.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        index = []
        for i in range(len(sort_ids)):
            if stage == "train":
                # 对于 train 来说要排除自己
                index.append(sort_ids[i][1:number + 1])
            else:
                index.append(sort_ids[i][0:number])

        if stage == "train":
            self.history.append(index)

        return index

    def get_history(self):
        temp = self.history
        self.history = []
        return temp

    def update(self, index, vectors):
        for id, vector in zip(index, vectors):
            self.vector[id] = vector


logger = logging.getLogger(__name__)

MODEL_CLASSES = (RobertaConfig, RobertaModel, RobertaTokenizer)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    if hasattr(args, 'beam_size'):
        beam_size = args.beam_size
    else:
        beam_size = 10
    if hasattr(args, 'max_target_length'):
        max_target_length = args.max_target_length
    else:
        max_target_length = 64

    # load generator
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    generator = Generator(encoder=encoder, decoder=decoder, config=config,
                          beam_size=args.beam_size, max_length=args.max_target_length,
                          sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    generator._set(tokenizer, beam_size, max_target_length)
    if args.load_generator_from_checkpoint:
        checkpoint_prefix = f'checkpoint-best-bleu/{args.checkpoint_generator_name}'
        path = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        generator.load_state_dict(torch.load(path))
        logger.info(f"*** Loading generator from {path} ***")

    # load retriever
    retriever = Retriever.from_pretrained(args.model_name_or_path)
    retriever._set(tokenizer)
    if args.load_retriever_from_checkpoint:
        checkpoint_prefix = f'checkpoint-best-bleu/{args.checkpoint_retriever_name}'
        path = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        retriever.load_state_dict(torch.load(path))
        logger.info(f"*** Loading retriever from {path} ***")

    logger.info("Finish loading generator [%s] from %s", get_model_size(generator), args.model_name_or_path)
    logger.info("Finish loading retriever [%s] from %s", get_model_size(retriever), args.model_name_or_path)

    return config, generator, retriever, tokenizer


class Generator(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self,
                 encoder: RobertaModel,
                 decoder: nn.TransformerDecoder,
                 config: RobertaConfig,
                 beam_size=None,
                 max_length=None,
                 sos_id=None,
                 eos_id=None):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _set(self, tokenizer: RobertaTokenizer, beam_size: int, max_length: int):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                decoder_attention_mask=None,
                score=None,
                is_generate=False):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        encoder_output = encoder_outputs[0].permute([1, 0, 2]).contiguous()
        if not is_generate:
            attn_mask = -1e4 * (1 - self.bias[:labels.shape[1], :labels.shape[1]])
            tgt_embeddings = self.encoder.embeddings(labels).permute([1, 0, 2]).contiguous()
            decoder_outputs = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                                           memory_key_padding_mask=(~attention_mask).bool())
            hidden_states = torch.tanh(self.dense(decoder_outputs)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)

            active_loss = decoder_attention_mask[..., 1:].ne(0)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable PP
                shift_labels = shift_labels.to(lm_logits.device)
                loss = None
                for idx, (logit, label, active) in enumerate(zip(shift_logits, shift_labels, active_loss)):
                    _loss = loss_fct(logit.view(-1, logit.size(-1))[active],
                                     label.view(-1)[active])
                    if score is not None:
                        _loss = _loss * score[idx]

                    if loss is None:
                        loss = _loss
                    else:
                        loss += _loss

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(input_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = attention_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.get_current_state()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(~context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.get_current_origin()))
                    input_ids = torch.cat((input_ids, beam.get_current_state()), -1)
                hyp = beam.get_hyp(beam.get_final())
                pred = beam.build_target_tokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            # suppose beam size is 1
            return_preds = torch.cat(preds, 0).squeeze(dim=1)
            return return_preds


class Retriever(RobertaModel):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)

    def _set(self, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids: Optional[torch.LongTensor] = None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = super(Retriever, self).forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return nn.functional.normalize(hidden_state, p=2, dim=1)


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def get_current_state(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode='floor')
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def get_final(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def get_hyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def build_target_tokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
