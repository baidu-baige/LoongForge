#!/bin/bash

set -euo pipefail

echo "========= build enter ========="

echo "$PATH"
WORK_DIR=$(cd $(dirname $0) && pwd) && cd $WORK_DIR

echo_cmd() {
    echo $1
    $1
}

echo "========= build BaigeOmni ========="

echo_cmd "rm -rf output"
echo_cmd "mkdir -p output"

rm -rf output/.scm/
# NOTE: 适配icode项目名
tar -zcvf ../AIAK-Training-Omni.tar.gz ../AIAK-Training-Omni/
mv ../AIAK-Training-Omni.tar.gz ./output/

echo "========= build exit ========="