"""
This approach uses BLEU-4 to evaluate training
"""
from __future__ import absolute_import

from argparse import Namespace
import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from io import open
import torch.nn as nn
from model import DataBase, build_model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from evaluator.bleu import calc_bleu
from typing import List
from args_config import add_args

from transformers import (PreTrainedTokenizer, AdamW, get_linear_schedule_with_warmup)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename: str) -> List[Example]:
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="read examples"):
        source = row["source"]
        target = row["target"]
        examples.append(
            Example(
                idx=idx,
                source=source,
                target=target
            )
        )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 query_ids
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.query_ids = query_ids


def convert_examples_to_features(examples: List[Example], tokenizer: PreTrainedTokenizer, args: Namespace,
                                 stage: str = None):
    """convert examples to token ids"""
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples), desc="convert examples to features"):
        # source
        source_str = example.source
        source_tokens = tokenizer.tokenize(source_str)
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens[:args.query_length])
        query_tokens = [tokenizer.cls_token] + source_tokens[:args.query_length - 2] + [tokenizer.eos_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        padding_length = args.query_length - len(query_ids)
        query_ids += [tokenizer.pad_token_id] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_str = example.target
            target_tokens = tokenizer.tokenize(target_str)
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                query_ids
            )
        )
    return features


class MyDataset(Dataset):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def build_index(self, retriever, args):
        with torch.no_grad():
            inputs = [feature.query_ids for feature in self.features]
            inputs = torch.tensor(inputs, dtype=torch.long)
            dataset = TensorDataset(inputs)
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
            query_vecs = []
            retriever.eval()
            for batch in tqdm(dataloader, desc="build index", total=len(dataloader)):
                code_inputs = torch.tensor(batch[0]).to(args.device)
                code_vec = retriever(code_inputs)
                query_vecs.append(code_vec.cpu().numpy())
            query_vecs = np.concatenate(query_vecs, 0)
            index = DataBase(query_vecs)
        return index


def do_nothing_collator(batch):
    return batch


def main():
    args = add_args()

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logger_path = os.path.join(args.output_dir, 'train.log') if args.do_train else os.path.join(args.output_dir,
                                                                                                'test.log')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config, generator, retriever, tokenizer = build_model(args)

    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)
    retriever.to(args.device)
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
        retriever = torch.nn.DataParallel(retriever)

    prefix = [tokenizer.cls_token_id]
    postfix = [tokenizer.sep_token_id]
    # Use // to comment Java code
    sep = tokenizer.convert_tokens_to_ids(["\n", "//"])
    sep_ = tokenizer.convert_tokens_to_ids(["\n"])

    def cat_to_input(code, similar_assertion, similar_code):
        new_input = code + sep + similar_assertion + sep_ + similar_code
        new_input = prefix + new_input[:args.max_source_length - 2] + postfix
        padding_length = args.max_source_length - len(new_input)
        new_input += padding_length * [tokenizer.pad_token_id]
        return new_input

    def cat_to_output(assertion):
        output = prefix + assertion[:args.max_target_length - 2] + postfix
        padding_length = args.max_target_length - len(output)
        output += padding_length * [tokenizer.pad_token_id]
        return output

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_dataset = MyDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      collate_fn=do_nothing_collator)
        index = train_dataset.build_index(retriever, args)

        # Prepare optimizer and schedule (linear warmup and decay) for generator
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': retriever.parameters(), 'eps': 1e-8}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        len(train_dataloader) * args.num_train_epochs * 0.1),
                                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)

        args.max_steps = args.num_train_epochs * len(train_dataloader)
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
        logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.max_steps)

        patience, best_bleu, losses, eval_dataset = 0, 0, [], {}

        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for idx, batch in enumerate(bar):
                retriever.train()
                generator.train()
                query = [feature.query_ids for feature in batch]
                query = torch.tensor(query, dtype=torch.long).to(device)
                query_vec = retriever(query)
                query_vec_cpu = query_vec.detach().cpu().numpy()
                i = index.search(query_vec_cpu, args.passage_number, 'train')
                document = [train_dataset.features[idx].query_ids for idxs in i for idx in idxs]
                document = torch.tensor(document, dtype=torch.long).to(device)
                document_vec = retriever(document)
                document_vec = document_vec.view(len(batch), args.passage_number, -1)
                score = torch.einsum('bd,bpd->bp', query_vec, document_vec)
                softmax = nn.Softmax(dim=-1)
                score = softmax(score)
                score = score.view(-1)

                # Cat the ids of code, relevant document to form the final input
                inputs, outputs = [], []
                for no, feature in enumerate(batch):
                    for j in i[no]:
                        relevant = train_dataset.features[j]
                        inputs.append(cat_to_input(feature.source_ids, relevant.target_ids, relevant.source_ids))
                        outputs.append(cat_to_output(feature.target_ids))
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                outputs = torch.tensor(outputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                target_mask = outputs.ne(tokenizer.pad_token_id)
                results = generator(input_ids=inputs, attention_mask=source_mask,
                                    labels=outputs, decoder_attention_mask=target_mask, score=score)
                loss = results.loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    # Update Vector
                    with torch.no_grad():
                        retriever.eval()
                        history = index.get_history()
                        for i in history:
                            document = [train_dataset.features[idx].query_ids for idxs in i for idx in idxs]
                            document = torch.tensor(document, dtype=torch.long).to(device)
                            document_vec = retriever(document)
                            update_ids = [id for j in i for id in j]

                            index.update(update_ids, document_vec.cpu().numpy())

                        retriever.train()

                bar.set_description("epoch {} loss {}".format(epoch, round(loss.item(), 3)))

            logger.info(score.view(-1, args.passage_number))

            if args.do_eval:
                index = train_dataset.build_index(retriever, args)
                # Eval model with dev dataset
                if 'dev_bleu' in eval_dataset:
                    eval_examples, eval_data = eval_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    eval_data = MyDataset(eval_features)
                    eval_dataset = (eval_examples, eval_data)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                             collate_fn=do_nothing_collator)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                retriever.eval()
                generator.eval()
                p = []
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Running Evaluation"):
                        query = [feature.query_ids for feature in batch]
                        query = torch.tensor(query, dtype=torch.long).to(device)
                        query_vec = retriever(query)
                        query_vec_cpu = query_vec.detach().cpu().numpy()
                        i = index.search(query_vec_cpu, 1)
                        inputs = []
                        for no, feature in enumerate(batch):
                            relevant = train_dataset.features[i[no][0]]
                            inputs.append(cat_to_input(feature.source_ids, relevant.target_ids, relevant.source_ids))

                        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                        source_mask = inputs.ne(tokenizer.pad_token_id)
                        preds = generator(inputs,
                                          attention_mask=source_mask,
                                          is_generate=True)
                        top_preds = list(preds.cpu().numpy())
                        p.extend(top_preds)
                p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
                generator.train()
                retriever.train()
                predictions, refs = [], []
                with open(args.output_dir + f"/dev_{epoch}.output", 'w') as f, open(
                        args.output_dir + f"/dev_{epoch}.gold", 'w') as f1:
                    for pred, gold in zip(p, eval_examples):
                        predictions.append(pred.strip())
                        refs.append([gold.target.strip()])
                        f.write(str(gold.idx) + '\t' + pred + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                dev_bleu = calc_bleu(refs, predictions)
                em = [int(ref[0] == prediction) for ref, prediction in zip(refs, predictions)]
                logger.info("  ******* Eval Result *******")
                logger.info("  %s = %s " % ("loss", str(round(losses[-1], 5))))
                logger.info("  %s = %s " % ("em", str(sum(em) / len(em))))
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)

                # Test at each epoch
                train_examples = read_examples(args.train_filename)
                train_features = convert_examples_to_features(train_examples, tokenizer, args)
                train_dataset = MyDataset(train_features)
                index = train_dataset.build_index(retriever, args)

                test_examples = read_examples(args.test_filename)
                test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
                test_data = MyDataset(test_features)

                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size,
                                             collate_fn=do_nothing_collator)

                logger.info("***** Running Testing *****")
                logger.info("  Num examples = %d", len(test_examples))
                logger.info("  Batch size = %d", args.test_batch_size)

                retriever.eval()
                generator.eval()
                p = []
                for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Running Testing"):
                    query = [feature.query_ids for feature in batch]
                    query = torch.tensor(query, dtype=torch.long).to(device)
                    query_vec = retriever(query)
                    query_vec_cpu = query_vec.detach().cpu().numpy()
                    i = index.search(query_vec_cpu, 1)
                    inputs = []
                    for no, feature in enumerate(batch):
                        relevant = train_dataset.features[i[no][0]]
                        inputs.append(cat_to_input(feature.source_ids, relevant.target_ids, relevant.source_ids))
                    with torch.no_grad():
                        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                        source_mask = inputs.ne(tokenizer.pad_token_id)
                        preds = generator(inputs,
                                          attention_mask=source_mask,
                                          is_generate=True)
                        top_preds = list(preds.cpu().numpy())
                        p.extend(top_preds)
                p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
                generator.train()
                retriever.train()
                predictions, refs = [], []
                with open(args.output_dir + f"/test_{epoch}.output", 'w') as f, open(args.output_dir + f"/test_{epoch}.gold", 'w') as f1:
                    for pred, gold in zip(p, test_examples):
                        predictions.append(pred.strip())
                        refs.append([gold.target.strip()])
                        f.write(str(gold.idx) + '\t' + pred + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                match = [int(pred == ref[0]) for pred, ref in zip(predictions, refs)]
                test_bleu_score = calc_bleu(refs, predictions)
                logger.info("  ******* Test Result *******")
                logger.info("  %s = %s " % ("em: ", str(sum(match) / len(predictions))))
                logger.info("  %s = %s " % ("bleu-4", str(test_bleu_score)))
                logger.info("  " + "*" * 20)

                # Save best bleu retriever and generator
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = generator.module if hasattr(generator,
                                                                'module') else generator  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "generator_model.bin")
                    if os.path.exists(output_model_file):
                        os.remove(output_model_file)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save = retriever.module if hasattr(retriever,
                                                                'module') else retriever  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "retriever_model.bin")
                    if os.path.exists(output_model_file):
                        os.remove(output_model_file)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience = 0
                else:
                    # early stop if bleu not increase in two epoch
                    patience += 1
                    if patience == 2:
                        logger.info("two epoches passed after last saving but bleu-4 not increase, early stopped.")
                        break

    if args.do_test:
        # load retriever
        checkpoint_prefix = 'checkpoint-best-bleu/retriever_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = retriever.module if hasattr(retriever, 'module') else retriever
        model_to_load.load_state_dict(torch.load(output_dir))

        # load generator
        checkpoint_prefix = 'checkpoint-best-bleu/generator_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = generator.module if hasattr(generator, 'module') else generator
        model_to_load.load_state_dict(torch.load(output_dir))

        # Build retrival base
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args)
        train_dataset = MyDataset(train_features)
        index = train_dataset.build_index(retriever, args)

        test_examples = read_examples(args.test_filename)
        test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
        test_data = MyDataset(test_features)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size,
                                     collate_fn=do_nothing_collator)

        logger.info("***** Running Testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)

        retriever.eval()
        generator.eval()
        p = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Running Testing"):
            query = [feature.query_ids for feature in batch]
            query = torch.tensor(query, dtype=torch.long).to(device)
            query_vec = retriever(query)
            query_vec_cpu = query_vec.detach().cpu().numpy()
            i = index.search(query_vec_cpu, 1)
            inputs = []
            for no, feature in enumerate(batch):
                relevant = train_dataset.features[i[no][0]]
                inputs.append(cat_to_input(feature.source_ids, relevant.target_ids, relevant.source_ids))
            with torch.no_grad():
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                preds = generator(inputs,
                                  attention_mask=source_mask,
                                  is_generate=True)
                top_preds = list(preds.cpu().numpy())
                p.extend(top_preds)
        p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
        generator.train()
        retriever.train()
        predictions, refs = [], []
        with open(args.output_dir + "/test.output", 'w') as f, open(args.output_dir + "/test.gold", 'w') as f1:
            for pred, gold in zip(p, test_examples):
                predictions.append(pred.strip())
                refs.append([gold.target.strip()])
                f.write(str(gold.idx) + '\t' + pred + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')

        match = [int(pred == ref[0]) for pred, ref in zip(predictions, refs)]
        test_bleu_score = calc_bleu(refs, predictions)
        logger.info("  %s = %s " % ("em: ", str(sum(match) / len(predictions))))
        logger.info("  %s = %s " % ("bleu-4", str(test_bleu_score)))
        logger.info("  " + "*" * 20)


if __name__ == '__main__':
    main()
