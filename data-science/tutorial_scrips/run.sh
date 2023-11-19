
python data-science/src/prep.py \
    --raw_data "data/taxi-data.csv" \
    --train_data "output" \
    --test_data "output"

python data-science/src/train.py \
    --train_data "output/train.parquet" \
    --model_output "output/model" 

python data-science/src/evaluate.py \
    --model_input "output/model" \
    --test_data "output/test.parquet" \
    --evaluation_output "output"