import ray
from ray import serve

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
class Downloader:
    def __init__(self):
        pass


@ray.remote
class Preprocessor:
    def __init__(self):
        pass


@ray.remote
class ResNetClassify:
    def __init__(self):
        self.model = resnet18(pretrained=True).eval()
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

    def forward(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))
        print("[1/3] Parsed image data: {}".format(pil_image))

        pil_images = [pil_image]  # Our current batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        )
        print(
            "[2/3] Images transformed, tensor shape {}".format(
                input_tensor.shape
            )
        )

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        print("[3/3] Inference done!")
        return {"class_index": int(torch.argmax(output_tensor[0]))}


@ray.remote
class Hate:
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
    # ray_logo_bytes = requests.get(
    #     "https://ichef.bbci.co.uk/news/1024/branded_news/0D9B/production/_88738430_pic1go.jpg"
    # ).content

    # model = ModelOne.bind()
    # model_2 = ModelTwo.bind()
    # dag = model_2.forward.bind()
    # print(model_2)
    # dag = model.forward.bind(ray_logo_bytes)
    # print(ray.get(dag.execute()))
    config = [
        "config=projects/pythia/configs/textvqa/defaults.yaml",
        # "datasets=textvqa",
        "model=pythia",
        "run_type=test",
        "checkpoint.resume_zoo=/Users/jiaodong/Workspace/mmf/checkpoint/pythia_pretrained_vqa2.pth"
    ]
    model_2_actor = ModelTwo.remote("args")
    print(
        ray.get(
            model_2_actor.forward.remote(
                "https://ichef.bbci.co.uk/news/1024/branded_news/0D9B/production/_88738430_pic1go.jpg",
                "What is it",
            )
        )
    )