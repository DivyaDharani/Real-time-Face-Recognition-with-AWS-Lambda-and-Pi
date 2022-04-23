from torch import nn, optim, as_tensor
#from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
#from torch.optim import lr_scheduler
#from torch.nn.init import *
#from torchvision import transforms, utils, datasets, models
from models.inception_resnet_v1 import InceptionResnetV1

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

def build_model(num_classes):
    model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes = num_classes)
    last_conv_block = list(model_ft.children())[-6:]
    # print(last_conv_block)

    # Remove the last layers after conv block and place in layer_list .
    layer_list = list(model_ft.children())[-5:] # all final layers
    # print(f"layer_list: {layer_list}")

    # Put all beginning layers in an nn.Sequential . model_ft is now a torch model but without the final linear, pooling, batchnorm, and sigmoid layers.
    model_ft = nn.Sequential(*list(model_ft.children())[:-5])
    for param in model_ft.parameters():
        param.requires_grad = False

    # Then you can apply the final layers back to the new Sequential model.
    model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    model_ft.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=512, bias=False),
        normalize()
    )
    # model_ft.logits = nn.Linear(layer_list[3].in_features, len(class_names))
    model_ft.logits = nn.Linear(512, num_classes)
    model_ft.softmax = nn.Softmax(dim=1)
    return model_ft