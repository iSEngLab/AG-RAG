train_data_file="../../dataset/OldDataSet/assert_train_old.csv"
test_data_file="../../dataset/OldDataSet/assert_test_old.csv"
val_data_file="../../dataset/OldDataSet/assert_val_old.csv"

train_batch_size=8
eval_batch_size=8
test_batch_size=4

result_file_path="./result/OldDataSet/no_retriever_result.txt"
pred_file_path="./result/OldDataSet/no_retriever_prediction.csv"

CUDA_VISIBLE_DEVICES=1 python run-no-retriever.py \
    --model_name_or_path ../../codet5-base \
    --output_dir=./saved_models/OldDataSet_no_retriever \
    --do_train \
    --do_test \
    --do_eval \
    --train_data_file=${train_data_file} \
    --test_data_file=${test_data_file} \
    --eval_data_file=${val_data_file} \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --test_batch_size ${test_batch_size} \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --n_gpu 1 \
    --evaluate_during_training \
    --num_beams 1 \
    --result_file_path ${result_file_path} \
    --pred_file_path ${pred_file_path} \
    --seed 123456