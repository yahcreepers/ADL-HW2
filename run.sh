context=$1
test_file=$2
pred=$3
python preprocess.py --do_predict --context $context --test $test_file

python run_mc.py --model_name_or_path ./MC \
--output_dir out_MC \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--max_seq_length 384 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--overwrite_output_dir \
--do_predict \
--test_file test_pro.json

python run_qa.py --model_name_or_path ./QA \
--output_dir out_QA \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--max_seq_length 384 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--overwrite_output_dir \
--do_predict \
--test_file test_pro.json

python postprocess.py --file ./out_QA/predict_predictions.json --out $pred
