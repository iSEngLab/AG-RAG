from __future__ import absolute_import, division, print_function
import argparse
import torch
import logging
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import PreTrainedTokenizer
from embedding_model import build_model

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.input_ids = input_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        self.df = pd.read_csv(file_path)
        sources = self.df["source"].tolist()
        labels = self.df["target"].tolist()
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[
            i].decoder_input_ids, self.df["source"][i], self.df["target"][i]


def convert_examples_to_features(source, label, tokenizer: PreTrainedTokenizer, args) -> InputFeatures:
    # encode
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length',
                                  return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size,
                                         padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length',
                             return_tensors='pt')
    return InputFeatures(source_ids, label, decoder_input_ids)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", default=None, type=str,
                        help="The input file for embedding")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model embeddings output will be written")
    parser.add_argument("--output_file_name", default="embedding.csv", type=str,
                        help="The output file name")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_model_name", default=None, type=str,
                        help="the model name to load state dict")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for embedding")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--n_gpu', type=int, default=2,
                        help="using which gpu")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logger_path = os.path.join(args.output_dir, 'output.log')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    # Set seed
    set_seed(args)
    config, model, tokenizer = build_model(args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # start embedding
    dataset = TextDataset(tokenizer, args, file_path=args.file_path)
    # build dataloader
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0)
    # Test!
    logger.info("***** Running Embedding *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    result_map = {
        "source": [],
        "target": [],
        "embedding": []
    }
    bar = tqdm(dataloader, total=len(dataloader))
    for batch in bar:
        (input_ids, _, _, _, sources, targets) = [x.squeeze(1).to(args.device)
                                                  if type(x) is torch.Tensor else x for x in batch]
        with torch.no_grad():
            embeddings = model(input_ids)
        embeddings = embeddings.detach().cpu().tolist()
        for embedding, source, target in zip(embeddings, sources, targets):
            result_map["source"].append(source)
            result_map["target"].append(target)
            result_map["embedding"].append(embedding)

    pd.DataFrame(result_map).to_csv(os.path.join(args.output_dir, args.output_file_name), index=False)


if __name__ == "__main__":
    main()
