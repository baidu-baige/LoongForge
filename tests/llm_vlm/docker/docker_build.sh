# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

images_tag="dev_`date "+%Y%m%d_%H%M%S"`"
docker build -t registry.baidubce.com/hac-aiacc/LoongForge:$images_tag -f docker/Dockerfile .
docker push registry.baidubce.com/hac-aiacc/LoongForge:$images_tag
