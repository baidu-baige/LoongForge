images_tag="dev_`date "+%Y%m%d_%H%M%S"`"
docker build -t registry.baidubce.com/hac-aiacc/BaigeOmni:$images_tag -f docker/Dockerfile .
docker push registry.baidubce.com/hac-aiacc/BaigeOmni:$images_tag