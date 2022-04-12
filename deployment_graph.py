from io import BytesIO
from pydantic import BaseModel

import requests
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence

import ray
from ray import serve
from ray.experimental.dag.input_node import InputNode
from ray.serve.drivers import DAGDriver


# Model classes
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, word2idx, idx2word, idx):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.idx = idx

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[str(self.idx)] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, features):
        """Extract feature vectors from input images."""
        # with torch.no_grad():
        #     features = self.resnet(images)
        features = torch.Tensor(features.copy())
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

#############
# Serve Graph Nodes

class ContentInput(BaseModel):
    image_url: str
    user_id: int

@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
def download(inp: "ContentInput"):
    """Download HTTP content, in production this can be business logic downloading from other services"""
    return requests.get(inp.image_url).content

@serve.deployment
class Preprocessor:
    """Image preprocessor with imagenet normalization."""
    def __init__(self):
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
                # pixel values must be in the range [0,1] and we must then
                # normalize the image by the mean and standard deviation
                # of the ImageNet images' RGB channels.
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_payload_bytes) -> ray.ObjectRef:
        pil_image = Image.open(BytesIO(image_payload_bytes)).convert('RGB')
        input_tensor = self.preprocessor(pil_image).unsqueeze(0)
        # Cache in plasma throughout duration of request
        return ray.put(input_tensor)


@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
class ImageClassification_ResNet:
    def __init__(self, version: int):
        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        self.version = version

        resnet_model = models.resnet152(pretrained=True).eval()
        self.head = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        self.tail = list(resnet_model.children())[-1]


    def forward(self, input_tensor):
        # input_tensor = ray.get(input_tensor_objectref)
        with torch.no_grad():
            feat = self.head(input_tensor)
            output_tensor = self.tail(feat.reshape(feat.size(0), -1))


        probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        classify_result = []
        for i in range(top5_prob.size(0)):
            classify_result.append((self.categories[top5_catid[i]], top5_prob[i].item()))

        return {
            "classify_result": classify_result,
            "model_version": self.version,
            "last_layer_weights": feat.numpy(),
        }

@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
class ObjectDetection_MaskRCNN:
    def __init__(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

    def forward(self, input_tensor_objectref):
        input_tensor = ray.get(input_tensor_objectref)
        with torch.no_grad():
            output_boxes = self.model(input_tensor)
        return [{k: v.numpy() for k,v in box.items()} for box in output_boxes]

@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
class ImageCaption_ResNet_LSTM:
    def __init__(self):
        # Load vocabulary wrapper
        import json
        with open("data/word2idx.json") as word2idx_file, open("data/idx2word.json") as idx2word_file:
            word2idx = json.load(word2idx_file)
            idx2word = json.load(idx2word_file)
            self.vocab = Vocabulary(word2idx, idx2word, 9956)

        self.encoder = EncoderCNN(256).eval()
        self.decoder = DecoderRNN(256, 512, len(self.vocab), 1)

        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")

        self.encoder.load_state_dict(torch.load("models/encoder-5-3000.pkl"))
        self.decoder.load_state_dict(torch.load("models/decoder-5-3000.pkl"))

    def forward(self, mask_rnn_result, resnet_result):
        resnet_feature = resnet_result["last_layer_weights"]

        # Generate an caption from the image
        feature = self.encoder(resnet_feature)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[str(word_id)]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        return sentence

@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
class DynamicDispatch:
    def __init__(self, *handles):
        self.handles = handles

    async def forward(self, inp_tensor, inp: "ContentInput"):
        # selection logic
        chosen_idx = inp.user_id % len(self.handles)
        chosen_handle = self.handles[chosen_idx]
        return await chosen_handle.forward.remote(inp_tensor)

@serve.deployment(
    ray_actor_options={
        "num_cpus": 0.5
    }
)
def combine(resnet, mask_r_cnn, captioning):
    return {
        "captioning": captioning,
        "resnet_version": resnet["model_version"],
        # "mask_r_cnn": mask_r_cnn
        # no resnet and mask result direclty, too many nubmers and have json serde issue.
    }


# TODO: return image here.
# https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
# Demo note: use browser tab, not docs pages. docs page only work for JSON

# def show_aggregated_result():
#     from torchvision.utils import draw_bounding_boxes
#     import torch
#     import numpy as np
#     import matplotlib.pyplot as plt

#     import torchvision.transforms.functional as F
#     from torchvision.transforms import ConvertImageDtype
#     from torchvision.transforms.functional import convert_image_dtype
#     image = ConvertImageDtype(torch.uint8)(input_tensor)

#     plt.rcParams["savefig.bbox"] = 'tight'

#     def show(imgs):
#         if not isinstance(imgs, list):
#             imgs = [imgs]
#         fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#         for i, img in enumerate(imgs):
#             img = img.detach()
#             img = F.to_pil_image(img)
#             axs[0, i].imshow(np.asarray(img))
#             axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     score_threshold = .8
#     dogs_with_boxes = [
#         draw_bounding_boxes(image, boxes=output_box['boxes'][output_box['scores'] > score_threshold], width=4)
#     ]
#     show(dogs_with_boxes)


# Let's build the DAG
def input_schema_from_args(image_url: str, user_id: int) :
    return ContentInput(image_url=image_url, user_id=user_id)

preprocessor = Preprocessor.bind()
resnets = [
    ImageClassification_ResNet.bind(version=i)
    for i in range(3)
]
resnet_dispatch = DynamicDispatch.bind(*resnets)
mask_rcnn = ObjectDetection_MaskRCNN.bind()
image_caption = ImageCaption_ResNet_LSTM.bind()

with InputNode() as inp:
    downloaded_image_bytes = download.bind(inp)
    input_tensor = preprocessor.preprocess.bind(downloaded_image_bytes)

    resnet_output = resnet_dispatch.forward.bind(input_tensor, inp)
    mask_rcnnoutput = mask_rcnn.forward.bind(input_tensor)
    image_caption_output = image_caption.forward.bind(mask_rcnnoutput, resnet_output)

    dag = combine.bind(resnet_output, mask_rcnnoutput, image_caption_output)

    serve_entrypoint = DAGDriver.bind(dag, input_schema="deployment_graph.input_schema_from_args")

if __name__ == "__main__":
    print("Started running DAG locally...")
    url = "https://miro.medium.com/max/1400/1*gMR3ezxjF46IykyTK6MV9w.jpeg"
    rst = ray.get(dag.execute(ContentInput(image_url=url, user_id=2)))
    print(rst)
