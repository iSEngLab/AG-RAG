#!/bin/bash

display(){
    echo "===================="
    echo $1
    echo "===================="
}

result(){
    if [ $1 -eq 0 ];then
        display "Finish"
    else
        display "$2"
        exit
    fi
}

number=$2
display "GPUs are : $1"
display "passage number is : ${number}"

display "Start training!"
python codebert_main.py \
	--do_train \
	--do_eval \
	--model_name_or_path ./codebert-base \
	--train_filename ../../dataset/OldDataSet/assert_train_old.csv \
	--dev_filename ../../dataset/OldDataSet/assert_val_old.csv \
	--output_dir saved_models/OldDataSet_${number} \
	--max_source_length 512 \
	--max_target_length 64 \
	--query_length 256 \
	--beam_size 1 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 20 \
	--passage_number $number \
	--GPU_ids $1
result $? "Training failed!"

display "Start prediction!"
python codebert_main.py \
	--do_test \
	--model_name_or_path ./codebert-base \
	--train_filename ../../dataset/OldDataSet/assert_train_old.csv \
	--test_filename ../../dataset/OldDataSet/assert_test_old.csv \
	--output_dir saved_models/OldDataSet_${number} \
	--max_source_length 512 \
	--max_target_length 64 \
	--query_length 256 \
	--beam_size 1 \
	--test_batch_size 4 \
	--GPU_ids $1
result $? "Prediction failed!"
