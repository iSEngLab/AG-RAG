# Two Birds with One Stone: Improving Retrieval-Augmented Deep Assertion Generation via Joint Optimization

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
├── dataset.tar.gz : dataset archive
├── args_config.py
├── evaluator
├── model.py
├── requirements.txt
├── run-old.sh : fine-tuned CodeT5 using AG-RAG in NewDataSet
└── run-new.sh : fine-tuned CodeT5 using AG-RAG in OldDataSet
├── run.py
```

## Fine-tuned model and data

You can download the model directly through this [link](https://drive.google.com/drive/folders/1u3lCDk-ediE20U0XUHyTEg2ektC-I3Hd?usp=sharing) for testing, or you can use the data given above to train and test yourself.

Raw data are in dataset.tar.gz, using commands below to unzip dataset:

```bash
tar -xzcf dataset.tar.gz
```

## Fine-tuned and test CodeT5 to get AG-RAG result

```bash
bash run-new.sh <gpu_ids> <passage_number>
# -- or OldDataSet --
bash run-old.sh <gpu_ids> <passage_number>
```

> `bash run-new.sh 0,1 5` will use GPU0 and GPU1 to fine-tuned CodeT5, and during training retriever will retrieve 5 examplars to calculate loss.

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

run the commands below to get RQ2 results:

```bash
# RQ2 contains no_retriever, IR_retriever, pretrained_retriever, finetuned_retriever
cd RQ2/xxx

bash run-new.sh
# -- or OldDataSet --
bash run-old.sh
```

## RQ3

run the commands below to get RQ3 results:

```bash
# RQ3 contains CodeBERT, GraphCodeBERT and UniXcoder
cd RQ3/xxx

bash run-new.sh <gpu_ids> <passage_number>
# -- or OldDataSet --
bash run-old.sh <gpu_ids> <passage_number>
```

