#!/bin/bash

display() {
    echo "===================="
    echo $1
    echo "===================="
}

dataset_type=$1

codebase_path="./dataset/${dataset_type}DataSet/embedding_train.csv"
output_dir="./result/${dataset_type}DataSet"
query_data_paths=("./dataset/${dataset_type}DataSet/embedding_train.csv",
                  "./dataset/${dataset_type}DataSet/embedding_test.csv",
                  "./dataset/${dataset_type}DataSet/embedding_val.csv")
output_result_paths=("./result/${dataset_type}DataSet/assert_train_${dataset_type}_pretrained_retriever.csv",
                    "./result/${dataset_type}DataSet/assert_test_${dataset_type}_pretrained_retriever.csv"
                    "./result/${dataset_type}DataSet/assert_val_${dataset_type}_pretrained_retriever.csv")

# hyperparameters
# calc batch size in one subprocess
batch_size=200
# similarity storage size
storage_size=2000
# subprocess count
cpu_count=20

for ((i=0; i<${#query_data_paths[@]}; i++)); do
    query_data_path="${query_data_paths[i]}"
    output_result_path="${output_result_paths[i]}"
    display "python run_retrieval.py --codebase_path ${codebase_path} --query_data_path ${query_data_path} --batch_size ${batch_size} --storage_size ${storage_size} --output_dir ${output_dir} --output_sim_path ${output_sim_path} --output_result_path ${output_result_path} --cpu_count ${cpu_count}"
    python run_retrieval.py \
      --codebase_path ${codebase_path} \
      --query_data_path ${query_data_path}  \
      --batch_size ${batch_size} \
      --storage_size ${storage_size}  \
      --output_dir ${output_dir} \
      --output_sim_path ${output_sim_path} \
      --output_result_path ${output_result_path}  \
      --cpu_count ${cpu_count}
done
