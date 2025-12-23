#!/bin/bash
set -eo pipefail

# 1、选择 testcase 的功能、test_function 还是 perf
# 2、确定 Node 数量和模型的分类集合，做遍历迭代的第一层入口
# 3、确定当前跑的所有模型列表
# 4、根据步骤2 确定每个Node 数量场景只需要创建一次容器, 新的Node 数量需要将以前的容器删除，创建新的容器来运行. (可能逻辑复杂点)
# 5、check 功能/性能正确性

SETCOLOR_SUCCESS="echo -en \\E[1;32m"
SETCOLOR_FAILURE="echo -en \\E[1;31m"
SETCOLOR_WARNING="echo -en \\E[1;33m"
SETCOLOR_NORMAL="echo -en  \\E[0;39m"

export task_created_flag="Created"
export task_running_flag="Running"
export task_finish_flag="Finished"

SUCCESS_echo(){
	$SETCOLOR_SUCCESS && echo "$1"  && $SETCOLOR_NORMAL
}

FAILURE_echo(){
	$SETCOLOR_FAILURE && echo "$1"  && $SETCOLOR_NORMAL
}

WARNING_echo(){
	$SETCOLOR_WARNING && echo "$1"  && $SETCOLOR_NORMAL
}

SUCCESS_echo_date(){
	$SETCOLOR_SUCCESS && echo $(date "+%Y-%m-%d_%H:%M.%S"):  "$1"  && $SETCOLOR_NORMAL
}

FAILURE_echo_date(){
	$SETCOLOR_FAILURE && echo $(date "+%Y-%m-%d_%H:%M.%S"):  "$1"  && $SETCOLOR_NORMAL
}

WARNING_echo_date(){
	$SETCOLOR_WARNING && echo $(date "+%Y-%m-%d_%H:%M.%S"):  "$1"  && $SETCOLOR_NORMAL
}


# 创建一个关联数组来存储 Node数量与Models 分类
declare -A categories

function save_node_and_model_categories(){
    # 遍历所有文件
    config_path=${scripts_root_path}/configs
    for file in $(ls $config_path); do
        if [[ $file == "common.yaml" ]]; then
            continue
        fi

        if [[ "$tasks" =~ "perf" ]]; then
            TOTAL_K8S_NODES_value=$(grep "TOTAL_K8S_NODES" "${config_path}/$file"| tail -n 1)
        else
            TOTAL_K8S_NODES_value=$(grep "TOTAL_K8S_NODES" "${config_path}/$file"| head -n 1)
        fi

        # 检查是否找到了TOTAL_K8S_NODES
        if [[ -z "$TOTAL_K8S_NODES_value" ]]; then
            FAILURE_echo_date "未找到 TOTAL_K8S_NODES 在文件 $file 中"
            exit -1
        else
            value=$(echo $TOTAL_K8S_NODES_value | awk -F': ' '{print $2}')
            # 将model_name添加到相应的分类中
            model_name="${file%.yaml}"
            categories[$value]+="$model_name "
        fi
    done
}


function check_env_ready(){
    if ! command -v kubectl &> /dev/null
    then
        FAILURE_echo_date "kubectl 二进制未在环境变量中找到, 请在 ipipe_start.sh 脚本中检查 kubectl_path 环境配置是否正确"
        exit -1
    fi

    if ! command -v kubectl-view-allocations &> /dev/null
    then
        FAILURE_echo_date "kubectl-view-allocations 二进制未在环境变量中找到, 请在 ipipe_start.sh 脚本中检查 kubectl_view_allocations_path 环境配置是否正确"
    fi
}

function check_pytorchjob_finish() {
    local pytorchjob_name=$1
    local namespace=$2
    local node_nums=$3
    local file_name=$4

    # 计数器和最长等待时间（以秒为单位）
    counter=0

    # 初始化最后一行变量
    last_line_log=""
    
    # 检查最长时间
    check_pytorchjob_timeout=${CHECK_PYTORCHJOB_TIMEOUT:-"86400"}

    # pod running 状态flag
    pod_running_flag=false

    # 调度的最大时长
    schedule_counter=0
    schedule_timeout=${SCHEDULE_TIMEOUT:-"300"}

    pod_logs=""

    # 循环检查文件内容
    while [ $counter -lt $check_pytorchjob_timeout ]; do
        # 检查pod running状态，等待running
        local pod_running_count=$(kubectl get pods -n ${namespace} |grep $pytorchjob_name |grep Running |wc -l)

        # todo ... 增加调度超时相关的检查
        if [[ "$pod_running_flag" == "false" ]] && [[ $pod_running_count -ne $node_nums ]]; then
            FAILURE_echo_date "pod $pytorchjob_name not running, 查询 events 事件信息"
            local pytorchjob_event=$(kubectl get event |grep $pytorchjob_name || echo "暂时没有 $pytorchjob_name 相关的事件信息")
            FAILURE_echo_date "$pytorchjob_event"
            # 休眠 1 秒
            sleep 1

            # 超过调度最大时间，并退出
            if [[ $schedule_counter -gt $schedule_timeout ]]; then
                FAILURE_echo_date "调度失败: 任务 ${pytorchjob_name} 超过 ${schedule_timeout}s 仍未 Running"

                local pytorchjob_desc=`kubectl -n ${namespace} describe pytorchjob ${pytorchjob_name}`
                WARNING_echo_date "该 PytorchJob 任务详情和事件信息如下:"
                echo "${pytorchjob_desc}"

                local gpu_resource_info=`kubectl-view-allocations -r gpu`
                echo ""
                WARNING_echo_date "当前集群 GPU 资源占用情况如下:"
                echo "${gpu_resource_info}"

                # 删除任务
                stop_pytorchjob ${file_name}

                # 退出指定的 110 状态码, 提供给外层调用方(流水线)处理其他逻辑, 比如发出调度失败通知等
                exit 110
            fi
            # 更新计数器
            counter=$((counter + 1))
            schedule_counter=$((schedule_counter + 1))
            continue
        else
            if [[ "$pod_running_flag" == "false" ]];then
                SUCCESS_echo_date "pod $pytorchjob_name is running"
                pod_running_flag=true
            fi
        fi

        master_pod_name_filter="master-0"
        select_pod_name_filter="${master_pod_name_filter}"
        if [[ $node_nums -gt 1 ]];then
            worker_pod_name_filter="worker-$((node_nums - 2))"
            select_pod_name_filter=${worker_pod_name_filter}
        fi

        select_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $select_pod_name_filter |awk '{print $1}'`
        pod_logs=`kubectl -n ${namespace} logs $select_pod_name -c pytorch || echo "Pod 仍未处于 Running 状态, 跳过此次查询log操作"`
        local latest_log=$(echo "$pod_logs"| tail -n 1)
        # 检查新行是否和最后一行不同
        if [ "$latest_log" != "$last_line_log" ]; then
            # 如果不同，打印新行并更新最后一行
            SUCCESS_echo_date "启动中, 最新[1]行日志: $latest_log"
            last_line_log=$latest_log
        fi

        # 检查pytorchjob 是否成功
        local pytorchjob_status=`kubectl -n ${namespace} get pytorchjob |grep $pytorchjob_name |awk '{print $2}'`
        if [[ "$pytorchjob_status" == "Succeeded" ]] || [[ "$pod_logs" =~ "Finish all jobs run ipipe from main.py" ]]; then
            SUCCESS_echo_date "模型训练完成, 查看完整日志:"
            SUCCESS_echo_date "$pod_logs"

            # 执行错误不吐给外层
            set +e
            # 将性能指标数据从容器中cp出来, 日志文件在"/workspace/logs" 目录下
            # cp master 的日志
            master_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $master_pod_name_filter |awk '{print $1}'`
            kubectl cp $namespace/$master_pod_name:logs ${scripts_root_path}/logs

            # cp worker 的日志
            if [[ $node_nums -gt 1 ]];then
                worker_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $worker_pod_name_filter |awk '{print $1}'`
                kubectl cp $namespace/$worker_pod_name:logs ${scripts_root_path}/logs
            fi

            set -e
            
            break
        fi

        # 检查pytorchjob 是否失败
        if [[ "$pytorchjob_status" == "Failed" ]] || [[ "$pod_logs" =~ "出现异常" ]] || [[ "$pod_logs" =~ "Traceback (most recent call last)" ]]; then
            FAILURE_echo_date "模型训练失败, PytorchJob 状态为Failed, 失败日志:"
            WARNING_echo_date "$pod_logs"
            # 执行错误不吐给外层
            set +e

            # 将性能指标数据从容器中cp出来, 日志文件在"/workspace/logs" 目录下
            # cp master 的日志
            master_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $master_pod_name_filter |awk '{print $1}'`
            kubectl cp $namespace/$master_pod_name:logs ${scripts_root_path}/logs

            # cp worker 的日志
            if [[ $node_nums -gt 1 ]];then
                worker_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $worker_pod_name_filter |awk '{print $1}'`
                kubectl cp $namespace/$worker_pod_name:logs ${scripts_root_path}/logs
            fi

            local pytorchjob_desc=`kubectl -n ${namespace} describe pytorchjob ${pytorchjob_name}`
            FAILURE_echo_date "该 PytorchJob 任务事件信息如下:"
            WARNING_echo_date "$pytorchjob_desc"

            set -e
            exit -1
        fi

        # 休眠 1 秒
        sleep 1

        # 更新计数器
        counter=$((counter + 1))
    done

    # 检查计数器是否超过了最长等待时间
    if [ $counter -eq $check_pytorchjob_timeout ]; then
        FAILURE_echo_date "超过最长等待时间, 模型训练任务没有正常启动或完成, 查看日志:"
        WARNING_echo_date "$pod_logs"
        exit 1
    fi
}

function run_pytorchjob() {
    local file_name=$1
    local pytorchjob_yaml_path=$2
    # 替换变量生成新的文件
    new_pytorcjob_yaml_path=${scripts_root_path}/${file_name}
    envsubst < ${pytorchjob_yaml_path} > ${new_pytorcjob_yaml_path}

    # 创建新的任务
    kubectl create -f ${new_pytorcjob_yaml_path}
}

function stop_pytorchjob() {
    local file_name=$1
    # 替换变量生成新的文件
    new_pytorcjob_yaml_path=${scripts_root_path}/${file_name}
    kubectl delete -f ${new_pytorcjob_yaml_path}

    sleep 15
}

function run_all_ipipe_case(){
    SUCCESS_echo_date "Begin run pipe case."

    check_env_ready

    save_node_and_model_categories ${scripts_root_path}

    # 遍历所有模型列表
    for node_nums in "${!categories[@]}";
    do
        SUCCESS_echo_date "node_nums: $node_nums"
        SUCCESS_echo_date "model_name: ${categories[$node_nums]}"

        local model_names="${categories[$node_nums]}"
        model_names=$(echo "$model_names" | awk '{$1=$1};1')

        # 判断只运行指定的模型；
        if [[ "${specific_model_name}" != "" ]]; then
            # convert the variables into arrays
            IFS=' ' read -r -a model_names_array <<< "$model_names"
            IFS=' ' read -r -a specific_model_name_array <<< "$specific_model_name"

            # find the intersection
            new_model_names=()
            for el in "${model_names_array[@]}"; do 
                for el2 in "${specific_model_name_array[@]}"; do
                    if [[ $el == $el2 ]]; then
                        new_model_names+=($el)
                    fi
                done
            done

            # print the intersection
            if [[ ${#new_model_names[@]} -eq 0 ]]; then
                WARNING_echo_date "不包含指定模型, 进入下一次循环."
                continue
            else
                model_names=${new_model_names[@]}
                SUCCESS_echo_date "包含运行指定模型 ${model_names}, 执行训练任务."
            fi
        fi
        
        pytorchjob_yaml_path=${scripts_root_path}/yaml_template/pytorchjob_standalone.yaml
        aiak_training_omni_folder="/workspace/AIAK-Training-Omni"

        # 透传变量给 run_pytorchjob 来生成yaml 文件使用, 使用方式：envsubst < ${pytorchjob_yaml_path} > ${new_pytorcjob_yaml_path}
        NAME_PREFIX=${NAME_PREFIX_AGILE:-"agile-aiak-transformer-run"}
        export NAMESPACE="default"
        export PYTORCHJOB_NAME=${NAME_PREFIX}-`date +%s`
        export IMAGE=${IMAGE:-"NA"}
        export TRAIN_DATA_DIR=${TRAIN_DATA_DIR}
        export GPU_RESOURCE=${GPU_RESOURCE}
        export GPU_COUNT=${GPU_NUMS}
        export WORKER_REPLICAS=$((node_nums - 1))
        export TIMEOUT=${TIMEOUT}
        export BOS_SYNC_AIAK_TRANSFORMER_ADDR=${BOS_SYNC_AIAK_TRANSFORMER_ADDR}
        export accuracy_relative_tolerance=${accuracy_relative_tolerance}
        export performance_relative_tolerance=${performance_relative_tolerance}
        export tasks=${tasks}
        export use_nccl=${use_nccl}
        export training_type=${training_type}

        if [[ $node_nums -gt 1 ]]; then
            pytorchjob_yaml_path=${scripts_root_path}/yaml_template/pytorchjob_distributed.yaml
        fi

        # 运行 pytorchjob 任务
        default_command=$(cat << EOF

                #! /bin/bash
                set -euo pipefail
                mkdir -p /workspace/logs
                
                echo "开始下载 aiak_training_omni"
                cd /workspace && rm -rf AIAK-Training-Omni
                wget ${BOS_SYNC_AIAK_TRANSFORMER_ADDR}
                tar -zxvf AIAK-Training-Omni.tar.gz
                echo "下载完成 aiak_training_omni"

                cd $aiak_training_omni_folder/tests
                extra_param="--node_nums ${node_nums} \
                             --gpu_nums ${gpu_nums} \
                             --models ${model_names} \
                             --tasks ${tasks} \
                             --accuracy_relative_tolerance ${accuracy_relative_tolerance} \
                             --performance_relative_tolerance ${performance_relative_tolerance} \
                             --training_type ${training_type} \
                             --timeout ${TIMEOUT}"
                if [[ "${use_nccl}" == "true" ]]; then
                    extra_param="\$extra_param --use_nccl"
                fi
                command="python3 main.py \$extra_param"
                echo "任务开始执行: \$command"
                eval \$command
EOF
)
        export PYTORCHJOB_COMMAND=${SPECIFIC_PYTORCHJOB_COMMAND:-${default_command}}
        file_name=pytorchjob_${node_nums}.yaml
        run_pytorchjob $file_name $pytorchjob_yaml_path

        # 判断创建的容器是否处于running状态, 查看容器日志是否符合预期
        check_pytorchjob_finish $PYTORCHJOB_NAME $NAMESPACE $node_nums $file_name

        # 删除任务
        stop_pytorchjob $file_name

    done

    SUCCESS_echo_date "End run pipe case."
}

