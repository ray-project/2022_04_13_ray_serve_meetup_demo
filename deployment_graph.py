import ray
from ray import ObjectRef, serve

import torch
from torchvision import transforms
from torchvision.models import resnet18

import mmf
from mmf.utils.inference import Inference
from omegaconf import OmegaConf
from mmf.utils.configuration import Configuration

from io import BytesIO
from PIL import Image

import requests


@ray.remote
def download(url):
    return requests.get(url).content

@ray.remote
class Preprocessor:
    def __init__(self):
        from torchvision import transforms

        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_payload_bytes) -> ObjectRef:
        pil_image = Image.open(BytesIO(image_payload_bytes))
        pil_images = [pil_image]  # Our current batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        )
        # Cache in plasma throughout duration of request
        return ray.put(input_tensor)


@ray.remote
class ResNetClassify:
    def __init__(self):
        self.model = resnet18(pretrained=True).eval()

    def forward(self, input_tensor_objectref):
        input_tensor = ray.get(input_tensor_objectref)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return {"class_index": int(torch.argmax(output_tensor[0]))}


@ray.remote
class HatefulMeme:
    def __init__(self) -> None:
        from mmf.models.mmbt import MMBT
        self.model = MMBT.from_pretrained("mmbt.hateful_memes.images")

    def forward(self, image_payload_bytes, query):
        from mmf.common.sample import Sample, SampleList

        image_url = "https://i.imgur.com/tEcsk5q.jpg" #@param {type:"string"}
        text = "look how many people love you" #@param {type: "string"}

        return self.model.classify(image_url, text)


@ray.remote
class Model3:
    pass


@ray.remote
class Composition:
    pass


if __name__ == "__main__":
    print(1)
    # Return image with metadata
    url = "https://ichef.bbci.co.uk/news/1024/branded_news/0D9B/production/_88738430_pic1go.jpg"

    downloaded_image_bytes = download.bind(url)
    preprocessor = Preprocessor.bind()
    input_tensor_objectref = preprocessor.preprocess.bind(downloaded_image_bytes)
    resnet = ResNetClassify.bind()
    dag = resnet.forward.bind(input_tensor_objectref)
    print(ray.get(dag.execute()))
    # model = ModelOne.bind()
    # model_2 = ModelTwo.bind()
    # dag = model_2.forward.bind()
    # print(model_2)
    # dag = model.forward.bind(ray_logo_bytes)
    # print(ray.get(dag.execute()))
    # config = [
    #     "config=projects/pythia/configs/textvqa/defaults.yaml",
    #     # "datasets=textvqa",
    #     "model=pythia",
    #     "run_type=test",
    #     "checkpoint.resume_zoo=/Users/jiaodong/Workspace/mmf/checkpoint/pythia_pretrained_vqa2.pth"
    # ]
    # model_2_actor = ModelTwo.remote("args")
    # print(
    #     ray.get(
    #         model_2_actor.forward.remote(
    #             "https://ichef.bbci.co.uk/news/1024/branded_news/0D9B/production/_88738430_pic1go.jpg",
    #             "What is it",
    #         )
    #     )
    # )
