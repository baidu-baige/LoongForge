#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}/..
source ${scripts_root_path}/common/common.sh

function summary_perf_metrics() {
    # Copy performance metric data from container, log file is in "/workspace/logs" directory
    metric_file=${scripts_root_path}/total_metric.csv
    echo "model_name,training_type_name,TP_PP_DP_ZERO,micro_batch_size,global_batch_size,seq_length,elapsed_time_per_iteration(ms),lr,initial_loss_scale,lm_loss,avg_throughput(tokens/device/s)" > $metric_file
    collection_model_flag=""

    # Retrieve logs where training type is pretrain
    model_training_logs=`ls ${scripts_root_path}/logs |grep training |grep run.log |grep pretrain`
    for model_training_log in ${model_training_logs};
    do
        # training log name example: training#llama-2-7b#pretrain#nodes_1#rank_0#run.log
        local model_name=`echo $model_training_log | awk -F'#' '{print $(2)}'`
        local training_type_name=`echo $model_training_log | awk -F'#' '{print $(3)}'`
        local nodes_nums=`echo $model_training_log | awk -F'#' '{print $(4)}' | cut -d'_' -f2`
        local RANK=`echo $model_training_log | awk -F'#' '{print $(5)}' | cut -d'_' -f2`

        if  [[ "${collection_model_flag}" =~ "${model_name}" ]];then
            continue
        fi

        collection_model_flag="${model_name},${collection_model_flag}"

        # RANK=0 represents master, RANK non-0 represents worker
        master_log_name="training#${model_name}#${training_type_name}#nodes_${nodes_nums}#rank_0#run.log"
        model_training_log_args_path="${scripts_root_path}/logs/${master_log_name}"
        metrics_model_training_log_path=${scripts_root_path}/logs/${master_log_name}
        if [[ "${nodes_nums}" != "1" ]];then
            worker_rank=$((nodes_nums - 1))
            worker_log_name="training#${model_name}#${training_type_name}#nodes_${nodes_nums}#rank_${worker_rank}#run.log"
            metrics_model_training_log_path=${scripts_root_path}/logs/${worker_log_name}
        fi

        args_file_path="${scripts_root_path}/logs/${model_name}_args.txt"

        sed -n "/------------------------ arguments ------------------------/,/end of arguments/p" ${model_training_log_args_path} > ${args_file_path}
        tensor_model_parallel_size=`grep '  tensor_model_parallel_size' $args_file_path | awk '{print $NF}'`
        pipeline_model_parallel_size=`grep '  pipeline_model_parallel_size' $args_file_path | awk '{print $NF}'`
        data_parallel_size=`grep '  data_parallel_size' $args_file_path | awk '{print $NF}'`
        micro_batch_size=`grep '  micro_batch_size' $args_file_path | awk '{print $NF}'`
        global_batch_size=`grep '  global_batch_size' $args_file_path | awk '{print $NF}'`
        seq_length=`grep '  seq_length' $args_file_path | awk '{print $NF}'`
        use_distributed_optimizer=`grep '  use_distributed_optimizer' $args_file_path | awk '{print $NF}'`
        lr=`grep '  lr ' $args_file_path | awk '{print $NF}'`

        initial_loss_scale=`grep '  initial_loss_scale' $args_file_path | awk '{print $NF}'`

        elapsed_time_per_iteration=`grep "elapsed time per iteration (ms):" $metrics_model_training_log_path |grep 'elapsed time per iteration (ms)' | tail -n 1 | awk 'BEGIN{FS="elapsed time per iteration \\\(ms\\\):";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}'`
        lm_loss=`grep "elapsed time per iteration (ms):" $metrics_model_training_log_path |grep 'lm loss' | tail -n 1  | awk 'BEGIN{FS="lm loss:";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}'`

        # tokens/device/s=(global_batch_size*seq_length)/total_time/num_cards
        through=`grep "throughput (token/sec/GPU):" $metrics_model_training_log_path |grep 'throughput (token/sec/GPU)' | tail -n 1 | awk 'BEGIN{FS="throughput \\\(token/sec/GPU\\\):";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}'`

        through_average_log=$(grep "throughput (token/sec/GPU):" $metrics_model_training_log_path | grep 'throughput (token/sec/GPU)' | awk 'BEGIN{FS="throughput \\(token/sec/GPU\\):";OFS=" "} {print $2}' | awk -F '|' '{print $1}' | awk '{print $1}' | tail -n +2 | awk '{ data[NR] = $1; sum += $1; n++ } END { formula = "(" data[1]; for(i=2; i<=n; i++) { formula = formula " + " data[i] }; formula = formula ") / " n " = "; average = (n > 0 ? sum / n : "N/A"); print formula average; }')

        through_average=$(grep "throughput (token/sec/GPU)" $metrics_model_training_log_path |grep 'throughput (token/sec/GPU)' | awk 'BEGIN{FS="throughput \\(token/sec/GPU\\):";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}' | tail -n +2 | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')

        SUCCESS_echo_date "------------------------ Performance Metrics Collection start ------------------------"
        SUCCESS_echo_date "Model: $model_name"
        SUCCESS_echo_date "Training Type: $training_type_name"
        SUCCESS_echo_date "pipeline_model_parallel_size: $pipeline_model_parallel_size"
        SUCCESS_echo_date "tensor_model_parallel_size: $tensor_model_parallel_size"
        SUCCESS_echo_date "data_parallel_size: $data_parallel_size"
        SUCCESS_echo_date "micro_batch_size: $micro_batch_size"
        SUCCESS_echo_date "global_batch_size: $global_batch_size"
        SUCCESS_echo_date "seq_length: $seq_length"
        SUCCESS_echo_date "elapsed_time_per_iteration: $elapsed_time_per_iteration"
        SUCCESS_echo_date "lr: $lr"
        SUCCESS_echo_date "initial_loss_scale: $initial_loss_scale"
        SUCCESS_echo_date "lm_loss: $lm_loss"
        SUCCESS_echo_date "through: $through"
        SUCCESS_echo_date "through_average: $through_average_log"


        echo "$model_name,${training_type_name},${tensor_model_parallel_size}_${pipeline_model_parallel_size}_${data_parallel_size}_${use_distributed_optimizer},$micro_batch_size,$global_batch_size,$seq_length,$elapsed_time_per_iteration,\"\"\"${lr}\"\"\",$initial_loss_scale,\"\"\"$lm_loss\"\"\",$through_average" >> $metric_file
        SUCCESS_echo_date "------------------------ Performance Metrics Collection end ------------------------"
    done
}

# Summarize performance data
summary_perf_metrics "$@"