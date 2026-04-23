#!/bin/bash
set -eo pipefail

# 1. Select functionality: testcase, test_function, or perf
# 2. Determine Node number and model collection, entry point for iteration
# 3. Determine the list of all models currently running
# 4. Determine container creation based on Node number from step 2; create once, delete old containers if new Node number required. (Logic might be complex)
# 5. Check function/performance correctness

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


# Create an associative array to store Node quantity and Models classification
declare -A categories

function save_node_and_model_categories(){
    # Traverse all files
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

        # Check if TOTAL_K8S_NODES was found
        if [[ -z "$TOTAL_K8S_NODES_value" ]]; then
            FAILURE_echo_date "TOTAL_K8S_NODES not found in file $file"
            exit -1
        else
            value=$(echo $TOTAL_K8S_NODES_value | awk -F': ' '{print $2}')
            # Add model_name to the corresponding classification
            model_name="${file%.yaml}"
            categories[$value]+="$model_name "
        fi
    done
}


function check_env_ready(){
    if ! command -v kubectl &> /dev/null
    then
        FAILURE_echo_date "kubectl binary not found in environment variables. Please check kubectl_path environment configuration in ipipe_start.sh script"
        exit -1
    fi

    if ! command -v kubectl-view-allocations &> /dev/null
    then
        FAILURE_echo_date "kubectl-view-allocations binary not found in environment variables. Please check kubectl_view_allocations_path environment configuration in ipipe_start.sh script"
    fi
}

function check_pytorchjob_finish() {
    local pytorchjob_name=$1
    local namespace=$2
    local node_nums=$3
    local file_name=$4

    # Counter and max wait time (in seconds)
    counter=0

    # Initialize last line variable
    last_line_log=""
    
    # Check max time
    check_pytorchjob_timeout=${CHECK_PYTORCHJOB_TIMEOUT:-"86400"}

    # pod running status flag
    pod_running_flag=false

    # Max schedule duration
    schedule_counter=0
    schedule_timeout=${SCHEDULE_TIMEOUT:-"300"}

    pod_logs=""

    # Loop to check file content
    while [ $counter -lt $check_pytorchjob_timeout ]; do
        # Check pod running status, wait for running
        local pod_running_count=$(kubectl get pods -n ${namespace} |grep $pytorchjob_name |grep Running |wc -l)

        # todo ... Add schedule timeout related checks
        if [[ "$pod_running_flag" == "false" ]] && [[ $pod_running_count -ne $node_nums ]]; then
            FAILURE_echo_date "pod $pytorchjob_name not running, querying events info"
            local pytorchjob_event=$(kubectl get event |grep $pytorchjob_name || echo "No event info for $pytorchjob_name currently")
            FAILURE_echo_date "$pytorchjob_event"
            # Sleep 1 second
            sleep 1

            # Exceed schedule max time, exit
            if [[ $schedule_counter -gt $schedule_timeout ]]; then
                FAILURE_echo_date "Schedule failed: Task ${pytorchjob_name} not Running after ${schedule_timeout}s"

                local pytorchjob_desc=`kubectl -n ${namespace} describe pytorchjob ${pytorchjob_name}`
                WARNING_echo_date "PytorchJob task details and event info as follows:"
                echo "${pytorchjob_desc}"

                local gpu_resource_info=`kubectl-view-allocations -r gpu`
                echo ""
                WARNING_echo_date "Current cluster GPU resource usage as follows:"
                echo "${gpu_resource_info}"

                # Delete task
                stop_pytorchjob ${file_name}

                # Exit with specific status code 110, for external caller (pipeline) to handle other logic, e.g., send schedule failure notification
                exit 110
            fi
            # Update counters
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
        pod_logs=`kubectl -n ${namespace} logs $select_pod_name -c pytorch || echo "Pod not in Running state yet, skipping this log query"`
        local latest_log=$(echo "$pod_logs"| tail -n 1)
        # Check if new line is different from last line
        if [ "$latest_log" != "$last_line_log" ]; then
            # If different, print new line and update last line
            SUCCESS_echo_date "Starting, latest [1] line log: $latest_log"
            last_line_log=$latest_log
        fi

        # Check if pytorchjob succeeded
        local pytorchjob_status=`kubectl -n ${namespace} get pytorchjob |grep $pytorchjob_name |awk '{print $2}'`
        if [[ "$pytorchjob_status" == "Succeeded" ]] || [[ "$pod_logs" =~ "Finish all jobs run ipipe from main.py" ]]; then
            SUCCESS_echo_date "Model training completed, view full log:"
            SUCCESS_echo_date "$pod_logs"

            # Do not throw execution errors to outer layer
            set +e
            # Copy performance metric data from container, log file is in "/workspace/logs" directory
            # cp master logs
            master_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $master_pod_name_filter |awk '{print $1}'`
            kubectl cp $namespace/$master_pod_name:logs ${scripts_root_path}/logs

            # cp worker logs
            if [[ $node_nums -gt 1 ]];then
                worker_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $worker_pod_name_filter |awk '{print $1}'`
                kubectl cp $namespace/$worker_pod_name:logs ${scripts_root_path}/logs
            fi

            set -e
            
            break
        fi

        # Check if pytorchjob failed
        if [[ "$pytorchjob_status" == "Failed" ]] || [[ "$pod_logs" =~ "Exception" ]] || [[ "$pod_logs" =~ "Traceback (most recent call last)" ]]; then
            FAILURE_echo_date "Model training failed, PytorchJob status is Failed, failure log:"
            WARNING_echo_date "$pod_logs"
            # Do not throw execution errors to outer layer
            set +e

            # Copy performance metric data from container, log file is in "/workspace/logs" directory
            # cp master logs
            master_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $master_pod_name_filter |awk '{print $1}'`
            kubectl cp $namespace/$master_pod_name:logs ${scripts_root_path}/logs

            # cp worker logs
            if [[ $node_nums -gt 1 ]];then
                worker_pod_name=`kubectl get pod -n ${namespace} |grep $pytorchjob_name |grep $worker_pod_name_filter |awk '{print $1}'`
                kubectl cp $namespace/$worker_pod_name:logs ${scripts_root_path}/logs
            fi

            local pytorchjob_desc=`kubectl -n ${namespace} describe pytorchjob ${pytorchjob_name}`
            FAILURE_echo_date "This PytorchJob task event info as follows:"
            WARNING_echo_date "$pytorchjob_desc"

            set -e
            exit -1
        fi

        # Sleep 1 second
        sleep 1

        # Update counter
        counter=$((counter + 1))
    done

    # Check if counter exceeded max wait time
    if [ $counter -eq $check_pytorchjob_timeout ]; then
        FAILURE_echo_date "Exceeded max wait time, model training task did not start or complete normally, view log:"
        WARNING_echo_date "$pod_logs"
        exit 1
    fi
}

function run_pytorchjob() {
    local file_name=$1
    local pytorchjob_yaml_path=$2
    # Replace variables to generate new file
    new_pytorcjob_yaml_path=${scripts_root_path}/${file_name}
    envsubst < ${pytorchjob_yaml_path} > ${new_pytorcjob_yaml_path}

    # Create new task
    kubectl create -f ${new_pytorcjob_yaml_path}
}

function stop_pytorchjob() {
    local file_name=$1
    # Replace variables to generate new file
    new_pytorcjob_yaml_path=${scripts_root_path}/${file_name}
    kubectl delete -f ${new_pytorcjob_yaml_path}

    sleep 15
}

function run_all_ipipe_case(){
    SUCCESS_echo_date "Begin run pipe case."

    check_env_ready

    save_node_and_model_categories ${scripts_root_path}

    # Iterate through all model lists
    for node_nums in "${!categories[@]}";
    do
        SUCCESS_echo_date "node_nums: $node_nums"
        SUCCESS_echo_date "model_name: ${categories[$node_nums]}"

        local model_names="${categories[$node_nums]}"
        model_names=$(echo "$model_names" | awk '{$1=$1};1')

        # Determine if only running specific models;
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
                WARNING_echo_date "Does not contain specific model, skipping to next loop."
                continue
            else
                model_names=${new_model_names[@]}
                SUCCESS_echo_date "Contains specific model ${model_names} to run, executing training task."
            fi
        fi
        
        pytorchjob_yaml_path=${scripts_root_path}/yaml_template/pytorchjob_standalone.yaml
        loongforge_folder="/workspace/LoongForge"

        # Pass variables to run_pytorchjob to generate yaml file, usage: envsubst < ${pytorchjob_yaml_path} > ${new_pytorcjob_yaml_path}
        NAME_PREFIX=${NAME_PREFIX_AGILE:-"agile-loongforge-transformer-run"}
        export NAMESPACE="default"
        export PYTORCHJOB_NAME=${NAME_PREFIX}-`date +%s`
        export IMAGE=${IMAGE:-"NA"}
        export TRAIN_DATA_DIR=${TRAIN_DATA_DIR}
        export GPU_RESOURCE=${GPU_RESOURCE}
        export GPU_COUNT=${GPU_NUMS}
        export WORKER_REPLICAS=$((node_nums - 1))
        export TIMEOUT=${TIMEOUT}
        export BOS_SYNC_Baige_TRANSFORMER_ADDR=${BOS_SYNC_Baige_TRANSFORMER_ADDR}
        export accuracy_relative_tolerance=${accuracy_relative_tolerance}
        export performance_relative_tolerance=${performance_relative_tolerance}
        export tasks=${tasks}
        export use_nccl=${use_nccl}
        export training_type=${training_type}

        if [[ $node_nums -gt 1 ]]; then
            pytorchjob_yaml_path=${scripts_root_path}/yaml_template/pytorchjob_distributed.yaml
        fi

        # Run pytorchjob task
        default_command=$(cat << EOF

                #! /bin/bash
                set -euo pipefail
                mkdir -p /workspace/logs
                
                echo "Start downloading loongforge"
                cd /workspace && rm -rf LoongForge
                wget ${BOS_SYNC_Baige_TRANSFORMER_ADDR}
                tar -zxvf LoongForge.tar.gz
                echo "Download complete loongforge"

                cd $loongforge_folder/tests
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
                echo "Task execution started: \$command"
                eval \$command
EOF
)
        export PYTORCHJOB_COMMAND=${SPECIFIC_PYTORCHJOB_COMMAND:-${default_command}}
        file_name=pytorchjob_${node_nums}.yaml
        run_pytorchjob $file_name $pytorchjob_yaml_path

        # Check if created container is in running state, check if container log is as expected
        check_pytorchjob_finish $PYTORCHJOB_NAME $NAMESPACE $node_nums $file_name

        # Delete task
        stop_pytorchjob $file_name

    done

    SUCCESS_echo_date "End run pipe case."
}

