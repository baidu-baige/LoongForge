#!/bin/bash
set -eo pipefail

# AGILE_MODULE_NAME="baidu/hac-aiacc/Megatron"
# AGILE_PIPELINE_NAME="ChangePipeline-main"
# hi_group=4249864
# hi_robot=db19f337899039eb2d225f25f99cfe2ce
# AGILE_PIPELINE_TRIGGER_USER=lijipeng01
# AGILE_COMPILE_BRANCH=main
# AGILE_COMMENTS='[{"Commit":"6d076e977738e9ff1d306c591b675cd0c4239a00","author":"lijipeng01","comment":"aiak-train-273【工程效能】Megatron 流水线增加多模型\n\nChange-Id: I1843b97273cd883e81266004dcd8e80e854fb4f5\n","committer":"lijipeng01"}]'

module="${AGILE_MODULE_NAME}"
PIPELINE_NAME="${AGILE_PIPELINE_NAME}"
hi_group=$hi_group
robot_token=$hi_robot
e2e_trigger_user="${AGILE_PIPELINE_TRIGGER_USER}"
pipeline_version="${AGILE_COMPILE_BRANCH}"
commit=`echo -e "${AGILE_COMMENTS}" | tr '\n' ' '`
pipeline_addr=https://console.cloud.baidu-int.com/devops/ipipe/workspaces/${AGILE_WORKSPACE_ID}/pipelines/${AGILE_PIPELINE_ID}/builds/list?branchName=${pipeline_version}
schedule_fail_reason="$1"
training_log_file_bos_addr="$2"
cluster_info="【集群名称: hwl-cce、集群ID: cce-e0isdmib】"

function sendUTMsg()
{
python <<EOF
# -*- coding: utf-8 -*-
#!coding=utf-8
import sys
import time
import urllib2
import json
import re
import HiRobotApi as HiRobotApi

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

commit = '${commit}'
print 'commit: ', commit

comment = ""
matches = re.findall(r'"comment":"(.*?)"(?=,|$)', commit)
for match in matches:
    if not ('Merge' in match) and re.search('.*-\\d+', match):
        comment = match.split("Change-Id:")[0].strip()
comment = comment.replace('"', "")
comment = comment.replace('\\n', ' ')

info = '#### 代码库 \`%s\` 流水线 \`%s\` 训练作业任务调度通知' % ('$module','$PIPELINE_NAME')
info += '\\\n #### 详细信息'
info += '\\\n> **提交描述:** %s' % (comment)
info += '\\\n> **分支名:** %s' % ('$pipeline_version')
info += '\\\n> **触发同学:** %s' % ('$e2e_trigger_user')
info += '\\\n> **集群环境:** %s' % ('$cluster_info')
info += '\\\n> **作业状态:** <font color=\\\"red\\\">**作业调度失败**</font>'
info += '\\\n> **失败原因:** %s' % ('${schedule_fail_reason}. 可能原因是无可调度资源/集群调度作业异常, 可从调度日志查看当前集群 GPU 资源占用情况. 👇👇👇') 
info += '\\\n> **调度日志:** %s' % ('${training_log_file_bos_addr}') 
info += '\\\n>- 建议1: %s' % ('如要正常运行, 请联系集群上占用资源的同学清理下不紧急的作业任务, 流水线中重新运行训练作业') 
info += '\\\n>- 建议2: %s' % ('如要跳过此卡关任务, 流水线重新运行-选择单选项【skip_test】为 true 跳过该阶段任务') 
info += '\\\n> **流水线地址:** %s' % ('$pipeline_addr')

HiRobotApi.pushInfo(info, toid='$hi_group', Hi_Robot_Access_Token='$robot_token')
HiRobotApi.pushATInfo("👆👆👆", toid='$hi_group', Hi_Robot_Access_Token='$robot_token', atuserid='$e2e_trigger_user')
EOF
}

sendUTMsg

# ➡️