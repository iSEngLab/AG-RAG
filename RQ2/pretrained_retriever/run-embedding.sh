file_path=$1
output_file_name=$2
dataset_type=$3

CUDA_VISIBLE_DEVICES=0 python run_embedding.py \
  --file_path $file_path \
  --output_dir ./result/${dataset_type}DataSet \
  --output_file_name $output_file_name \
  --model_name_or_path codet5-base \
  --encoder_block_size 512 \
  --decoder_block_size 256 \
  --batch_size 16 \
  --n_gpu 1 \
  --seed 123456