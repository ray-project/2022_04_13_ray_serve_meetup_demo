noop:
	echo "Doing nothing, please use `make actual_target`"

models/decoder-5-3000.pkl:
	mkdir -p models
	wget https://ray-serve-blog.s3.us-west-2.amazonaws.com/2022_04_13_ray_serve_meetup_demo/models/decoder-5-3000.pkl -O models/decoder-5-3000.pkl

models/encoder-5-3000.pkl:
	mkdir -p models
	wget https://ray-serve-blog.s3.us-west-2.amazonaws.com/2022_04_13_ray_serve_meetup_demo/models/encoder-5-3000.pkl -O models/encoder-5-3000.pkl

prep: models/decoder-5-3000.pkl models/encoder-5-3000.pkl
	python -c "import torchvision.models as models; models.resnet152(pretrained=True)"
	python -c "import torchvision.models as models; models.resnet50(pretrained=True)"
	python -c "import torchvision.models as models; models.detection.maskrcnn_resnet50_fpn(pretrained=True)"



