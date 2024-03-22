data_file="./NewDataSet/assert_test_new.csv"
result_file="./NewDataSet/test.output"
gold_file="./NewDataSet/test.gold"
output_file="./assertion_type_new.csv"

python calculate.py  \
  --data_file ${data_file}  \
  --result_file ${result_file}  \
  --gold_file ${gold_file} \
  --output_file ${output_file} \
  --dataset new