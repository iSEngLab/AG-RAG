train_data_file="../../dataset/NewDataSet/assert_train_new.csv"
test_data_file="../../dataset/NewDataSet/assert_test_new.csv"
val_data_file="../../dataset/NewDataSet/assert_val_new.csv"

train_batch_size=8
eval_batch_size=8
test_batch_size=4

result_file_path="./result/NewDataSet/no_retriever_result.txt"
pred_file_path="./result/NewDataSet/no_retriever_prediction.csv"

CUDA_VISIBLE_DEVICES=0 python run_no_retriever.py \
    --model_name_or_path codet5-base \
    --output_dir=./saved_models/NewDataSet_no_retriever \
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