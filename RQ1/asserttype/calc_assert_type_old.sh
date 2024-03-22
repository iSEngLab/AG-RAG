data_file="./OldDataSet/assert_test_old.csv"
result_file="./OldDataSet/test.output"
gold_file="./OldDataSet/test.gold"
output_file="./assertion_type_old.csv"

python calculate.py  \
  --data_file ${data_file}  \
  --result_file ${result_file}  \
  --gold_file ${gold_file} \
  --output_file ${output_file} \
  --dataset old