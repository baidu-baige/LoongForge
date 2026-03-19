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
# NOTE: icode repo name cannot change now
cp -r ../AIAK-Training-Omni ../BaigeOmni
tar -zcvf ../BaigeOmni.tar.gz ../BaigeOmni/
mv ../BaigeOmni.tar.gz ./output/

echo "========= build exit ========="