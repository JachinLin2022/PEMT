#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"
export http_proxy='http://127.0.0.1:7890'
export https_proxy='http://127.0.0.1:7890'

do_train=true
do_eval=true
do_test=true
warmup_steps=500
save_steps=1000
save_strategy="no"
model_name_or_path="t5-base"
tokenizer_name="t5-base"
save_total_limit=1
per_device_train_batch_size=128
per_device_eval_batch_size=128
load_best_model_at_end=true
metric_for_best_model="average_metrics"
greater_is_better=true
evaluation_strategy="epoch"
non_linearity="gelu_new"
max_source_length=512
learning_rate=5e-4
split_validation_test=true
dataset_config_name=("en")
eval_dataset_config_name=("en")
test_dataset_config_name=("en")
predict_with_generate=true
overwrite_output_dir=true
compute_memory=true
report_to="none"
add_lora=true
train_lora=true
add_task_embedding=true
init_task_from_vocab=true

big_task=(superglue-multirc)
small_task=(superglue-cb superglue-wsc-fixed)
qa_task=(nq newsqa searchqa hotpotqa) 
other_Task=(paws scitail winogrande yelp_polarity)

target_task=(paws scitail winogrande yelp_polarity)

for task in ${target_task[@]}
do

    t=($task)
    num_train_epochs=20
    warmup_steps=0
    per_device_train_batch_size=128
    max_source_length=256
    if [[ "${big_task[@]}" =~ "${task}" ]]; then
        num_train_epochs=10
        max_source_length=348
        per_device_train_batch_size=64
    fi

    if [[ "${small_task[@]}" =~ "${task}" ]]; then
        per_device_train_batch_size=32
    fi

    if [[ "${qa_task[@]}" =~ "${task}" ]]; then
        per_device_train_batch_size=64
        per_device_eval_batch_size=128
        max_source_length=512
        num_train_epochs=5
    fi

    if [[ "yelp_polarity" =~ "${task}" ]]; then
        num_train_epochs=5
        per_device_train_batch_size=100
    fi



    output_dir="/home/linzhisheng/ATTEMPT/attempt/result/stage1_no_prefix/"$task
    while true
    do
    echo $task $num_train_epochs
        python run_seq2seq.py \
        --do_train=$do_train \
        --do_eval=$do_eval \
        --do_test=$do_test \
        --warmup_steps=$warmup_steps \
        --save_steps=$save_steps \
        --save_strategy="$save_strategy" \
        --model_name_or_path="$model_name_or_path" \
        --tokenizer_name="$tokenizer_name" \
        --save_total_limit=$save_total_limit \
        --per_device_train_batch_size=$per_device_train_batch_size \
        --per_device_eval_batch_size=$per_device_eval_batch_size \
        --load_best_model_at_end=$load_best_model_at_end \
        --metric_for_best_model="$metric_for_best_model" \
        --greater_is_better=$greater_is_better \
        --evaluation_strategy="$evaluation_strategy" \
        --non_linearity="$non_linearity" \
        --max_source_length=$max_source_length \
        --learning_rate=$learning_rate \
        --output_dir="$output_dir" \
        --split_validation_test=$split_validation_test \
        --task_name="${t[@]}" \
        --eval_dataset_name="${t[@]}" \
        --test_dataset_name="${t[@]}" \
        --num_train_epochs=$num_train_epochs \
        --dataset_config_name="${dataset_config_name[@]}" \
        --eval_dataset_config_name="${eval_dataset_config_name[@]}" \
        --test_dataset_config_name="${test_dataset_config_name[@]}" \
        --predict_with_generate=$predict_with_generate \
        --overwrite_output_dir=$overwrite_output_dir \
        --compute_memory=$compute_memory \
        --report_to="$report_to" \
        --train_lora=$train_lora \
        --add_lora=$add_lora \
        --add_task_embedding=$add_task_embedding \
        --init_task_from_vocab=$init_task_from_vocab \
        --logging_steps 10
        if [[ $? -eq 0 ]]; then
            # 如果脚本正常退出，则跳出循环
            break
        fi
    done
    bash clean.sh $output_dir
done