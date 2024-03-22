#!/bin/bash

dataset_type=Old

# dataset embedding
input_filenames=("assert_train_old.csv" "assert_test_old.csv" "assert_val_old.csv")
output_filenames=("embedding_train.csv" "embedding_test.csv" "embedding_val.csv")

for ((i=0; i<${#input_filenames[@]}; i++)); do
    input_filename="${input_filenames[i]}"
    output_filename="${output_filenames[i]}"
    echo "bash run-embedding.sh ${input_filename} ${output_filename} ${dataset_type}"
    bash run-embedding.sh ${input_filename} ${output_filename} ${dataset_type}
done

# dataset retrieval
bash run-retrieval.sh ${dataset_type}

bash run-finetuned-retriever.sh ${dataset_type}