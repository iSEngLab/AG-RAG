# Two Birds with One Stone: Improving Retrieval-AugmentedDeep Assertion Generation via Joint Optimization

This is the official PyTorch implementation for the following ISSTA 2024 submission:

Title: Two Birds with One Stone: Improving Retrieval-AugmentedDeep Assertion Generation via Joint Optimization

## Environment Setup

```bash
conda env create --name AG-RAG python=3.9
conda activate AG-RAG
pip install -r requirements.txt
```

tips: torch version may depend on CUDA version, so you should check the version of CUDA and replace it in requirements.txt.

## Folder Structure

```bash
.
├── README.md
├── RQ1
│   ├── asserttype : The source code to get assert type result
│   ├── result : The result of RQ1
│   └── venn : The source code to generate venn diagrams
├── RQ2
│   ├── IR_retriever : The source code to get IR-retriever result
│   ├── finetuned_retriever : The source code to get finetuned-retriever result
│   ├── no_retriever: The source code to get no-retriever result
│   ├── pretrained_retriever : The source code to get pretrained-retriever result
│   └── split_result.py : aims to split result from xxx_result.txt to xxx.gold and xxx.output in RQ2
├── RQ3
│   ├── CodeBERT
│   ├── GraphCodeBERT
│   └── UniXcoder
├── d4j : The source code to run defects4j in AG-RAG
├── dataset.tar.gz : dataset archive
├── args_config.py
├── evaluator
├── model.py
├── requirements.txt
├── run-old.sh
├── run.py
└── run.sh
```

## Fine-tuned model and data

Due to the huge model size, we are unable to upload fine-tuned models to the anonymous website. All fine-tuned models will be available upon acception.

Raw data are in dataset.tar.gz, using commands below to unzip dataset:

```bash
tar -xzcf dataset.tar.gz
```

## Fine-tuned and test CodeT5 to get AG-RAG result

```bash
bash run.sh <gpu_ids> <passage_number>
```

> `bash run.sh 0,1 5` will use GPU0 and GPU1 to fine-tuned CodeT5, and during training retriever will retrieve 5 examplars to calculate loss.

## RQ1

### Metrics

run the commands below to metrics the result of AG-RAG:

```bash
python evaluator/acc.py <gold_file_path> <output_file_path>
python evaluator/bleu.py <gold_file_path> <output_file_path>
python evaluator/CodeBLEU/calc_code_bleu.py --refs <gold_file_path> --hyp <output_file_path> --lang java
```

### Assert Type

run the commands below to get assert type results:

```bash
cd RQ1/asserttype
bash calc_assert_type_new.sh
bash calc_assert_type_old.sh
```

### Venn Diagram

run the commands below to generate venn diagrams:

```bash
cd RQ1/venn
python to_excel.py
python generate_venn.py
```

## RQ2

TODO

## RQ3

TODO
