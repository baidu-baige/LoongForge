#!/bin/bash
set -eo pipefail

# AGILE_MODULE_NAME="baidu/hac-aiacc/Megatron"
# AGILE_PIPELINE_NAME="ChangePipeline-main"
# hi_group=4249864
# hi_robot=db19f337899039eb2d225f25f99cfe2ce
# AGILE_PIPELINE_TRIGGER_USER=lijipeng01
# AGILE_COMPILE_BRANCH=main
# AGILE_COMMENTS='[{"Commit":"6d076e977738e9ff1d306c591b675cd0c4239a00","author":"lijipeng01","comment":"aiak-train-273[Engineering Efficiency] Megatron pipeline multi-model support\n\nChange-Id: I1843b97273cd883e81266004dcd8e80e854fb4f5\n","committer":"lijipeng01"}]'

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
cluster_info="[Cluster Name: hwl-cce, Cluster ID: cce-e0isdmib]"

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

info = '#### Repository \`%s\` pipeline \`%s\` training job task scheduling notification' % ('$module','$PIPELINE_NAME')
info += '\\\n #### Details'
info += '\\\n> **Commit description:** %s' % (comment)
info += '\\\n> **Branch name:** %s' % ('$pipeline_version')
info += '\\\n> **Triggered by:** %s' % ('$e2e_trigger_user')
info += '\\\n> **Cluster environment:** %s' % ('$cluster_info')
info += '\\\n> **Job status:** <font color=\\\"red\\\">**Job scheduling failed**</font>'
info += '\\\n> **Failure reason:** %s' % ('${schedule_fail_reason}. Possible reasons: no schedulable resources/cluster scheduling job exception. Check the scheduling log for current cluster GPU resource usage. 👇👇👇') 
info += '\\\n> **Scheduling log:** %s' % ('${training_log_file_bos_addr}') 
info += '\\\n>- Suggestion 1: %s' % ('To run normally, please contact the person occupying resources on the cluster to clean up non-urgent job tasks, and re-run the training job in the pipeline') 
info += '\\\n>- Suggestion 2: %s' % ('To skip this blocking task, re-run pipeline and select the option [skip_test] to true to skip this stage task') 
info += '\\\n> **Pipeline address:** %s' % ('$pipeline_addr')

HiRobotApi.pushInfo(info, toid='$hi_group', Hi_Robot_Access_Token='$robot_token')
HiRobotApi.pushATInfo("👆👆👆", toid='$hi_group', Hi_Robot_Access_Token='$robot_token', atuserid='$e2e_trigger_user')
EOF
}

sendUTMsg

# ➡️