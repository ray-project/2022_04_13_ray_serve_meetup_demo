#%%
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

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
def downloader(inp: "ContentInput"):
    """Download HTTP content, in production this can be business logic downloading from other services"""
    return requests.get(inp.image_url).content

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
class Preprocessor:
    """Image preprocessor with imagenet normalization."""
    def __init__(self):
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
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


@serve.deployment(ray_actor_options={"num_cpus": 0.5})
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

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
class ObjectDetection_MaskRCNN:
    def __init__(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

    def forward(self, input_tensor_objectref):
        input_tensor = ray.get(input_tensor_objectref)
        with torch.no_grad():
            return self.model(input_tensor)

        # return [{k: v.numpy() for k,v in box.items()} for box in output_boxes]

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
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

    def forward(self, detection_result, classifier_result):
        resnet_feature = classifier_result["last_layer_weights"]

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

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
class DynamicDispatch:
    def __init__(self, *classifier_models):
        self.classifier_models = classifier_models

    async def forward(self, inp_tensor, inp: "ContentInput"):
        # selection logic
        chosen_idx = inp.user_id % len(self.classifier_models)
        chosen_model = self.classifier_models[chosen_idx]
        return await chosen_model.forward.remote(inp_tensor)

@serve.deployment(ray_actor_options={"num_cpus": 0.5})
def combine(image_ref, classify, object_detection, caption):
    # Cast to uint8 for showing segmentation
    image = ray.get(image_ref).squeeze().type(torch.uint8)
    return {
        "caption": caption,
        "resnet_version": classify["model_version"],
        "classify_result": classify["classify_result"],
        "image": image,
        "object_detection": object_detection
        # no resnet and mask result direclty, too many nubmers and have json serde issue.
    }


def show_segmentation(rst):
    inst_classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    import torch
    from torchvision.utils import draw_segmentation_masks

    image = rst["image"]
    detection_output = rst["object_detection"][0]

    dog1_masks = detection_output['masks']
    inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

    proba_threshold = 0.8
    bool_masks = detection_output['masks'] > proba_threshold

    # There's an extra dimension (1) to the masks. We need to remove it
    bool_masks = bool_masks.squeeze(1)

    import numpy as np
    import matplotlib.pyplot as plt

    import torchvision.transforms.functional as F

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    show(draw_segmentation_masks(image, bool_masks, alpha=0.7))


preprocessor = Preprocessor.bind()
classifiers = [
    ImageClassification_ResNet.bind(i) for i in range(3)
]
dynamic_dispatch = DynamicDispatch.bind(*classifiers)
object_detection = ObjectDetection_MaskRCNN.bind()
image_caption = ImageCaption_ResNet_LSTM.bind()

def input_adapter(image_url: str, user_id: int):
    return ContentInput(image_url=image_url, user_id=user_id)

# Let's build the DAG
with InputNode() as user_input:
    image_bytes = downloader.bind(user_input)
    image_tensor = preprocessor.preprocess.bind(image_bytes)
    classify_output = dynamic_dispatch.forward.bind(image_tensor, user_input)
    object_detection_output = object_detection.forward.bind(image_tensor)
    caption_output = image_caption.forward.bind(object_detection_output, classify_output)

    dag = combine.bind(image_tensor, classify_output, object_detection_output, caption_output)
    serve_entrypoint = DAGDriver.bind(dag, input_schema="deployment_graph.input_adapter")


if __name__ == "__main__":
    print("Started running DAG locally...")
    url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01833805_hummingbird.JPEG?raw=true"
    rst = ray.get(dag.execute(ContentInput(image_url=url, user_id=2)))
    print(rst)

    # show_segmentation(rst)
    # print(rst["caption"])
    # print(f"Classifier version: {rst['resnet_version']}")
    # for val in rst["classify_result"]:
    #     print(val)
