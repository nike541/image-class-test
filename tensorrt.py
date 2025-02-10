import torch 
import torchvision
import torch.onnx
import onnxruntime
import onnx
import cv2
import torch.nn as nn
from PIL import Image
# from efficientnet_pytorch import EfficientNet
import torchvision.models as models

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchsampler import ImbalancedDatasetSampler
import torchvision.models as models
import torchvision
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import time,os
from torch.optim import lr_scheduler
from torchmetrics import F1Score
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np


# from REID.CUSTOM_MODEL.models import MyViT

# from Mode import AMLConv

# def load_models(path):
#     model = AMLConv(10)

#     check_point  = torch.load(path)
#     model.load_state_dict(check_point["model_state_dict"])
#     return model

# def build_model():


#     model = models.efficientnet_b7(pretrained=True)
#     dummy_input = torch.randn(10, 3, 240, 240)

#     # print(*list(model.children())[-1])

#     model = torch.nn.Sequential(*list(model.children())[:-1])
#     # print(model)

#     # model = torch.nn.Sequential(*list(model().children())[:-1])

#     print(model(dummy_input).shape)

#     return model

# build_model()
# import sys
# # sys.exit()
class Network(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.classes = classes
        
#         self.backbone = nn.Sequential(*(list(list(models.efficientnet_b0(pretrained=True).children())[0].children())[:-1]))        
        self.backbone = nn.Sequential(*(list(models.efficientnet_b0(pretrained=True).children())[:-1]))
#         self.conv = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=7)
        # self.bn = nn.BatchNorm2d(160)
        self.fc = nn.Linear(in_features=1280, out_features=self.classes,bias=False)
        
    def forward(self,x):
        t = self.backbone(x)
#         t = self.conv(t)
        # t = self.bn(t)

        t = t.reshape(-1, 1280)
        
        # vec_norms = torch.linalg.vector_norm(t,dim=1)

        # vec_norms = torch.diagonal(torch.sqrt(t @ torch.t(t)))

        # wt_norms = torch.diagonal(torch.sqrt(self.fc.state_dict()['weight'] @ torch.t(self.fc.state_dict()['weight'])))

        # print(torch.t(self.fc.state_dict()['weight']).shape)
        # print(t.shape)

        return pairwise_cosine_similarity(t, self.fc.state_dict()['weight'])

        # numerator = t @ torch.t(self.fc.state_dict()['weight'])
        # vecs_l2 = torch.mul(t, t).sum(axis=1)
        # weights_l2 = torch.mul(self.fc.state_dict()['weight'], self.fc.state_dict()['weight']).sum(axis=1)
        # # print(vecs_l2.shape,weights_l2.shape)
        # denominator = torch.max(torch.sqrt(vecs_l2.reshape(-1,1) @ weights_l2.reshape(1,-1)), torch.tensor(1e-8))
        # t = torch.div(numerator, denominator)


        # wt_norms = torch.linalg.vector_norm(self.fc.state_dict()['weight'],dim=1)
        
        # norm_mat = torch.outer(vec_norms,wt_norms)
        
        # t = self.fc(t)/norm_mat
        
        # return t

def build_model():

    model = Network(2)
    print(model(torch.randn(1,3,128,224)).shape)

    return model

# build_model()
# import sys 
# sys.exit()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() #if tensor.requires_grad else tensor.cpu().numpy()


def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = '4' 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    print("Inputs files are the ", inputs)
   
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        # if (not input.has_dim_value()):
        #     continue;
        dim1 = input.type.tensor_type.shape.dim[0]

        print("this is dim 1 ",dim1)
        print("tHis is the actual value", dim1.dim_param)

        # update dim to be a symbolic value

        dim1.dim_param = sym_batch_dim
        print("This is after the chage las ",dim1.dim_param )
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim

    print("this is inputs  after ", inputs)

    outputs = model.graph.output

    print("Print this is outputs", outputs)


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

import torch.nn as nn

# def build_model(pretrained=True, fine_tune=True, num_classes=2):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.efficientnet_b0(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
       
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
        
#     # Change the final classification head.
#     model.classifier[1] =nn.Sequential(nn.Linear(1280,512),nn.Linear(in_features=512, out_features=num_classes))
    
#     return model

# model = load_models("/home/stiwari/Desktop/Unsupervised_Object_localisation/Additive_Margin_loss_model_train/models/Adms490.pth")

# print(model)

path = "/home/iw/Downloads/ev_nev/efficient_net_20n.pth"


model = torch.load(path)

# print(model.keys())

models_eff = build_model()
models_eff.load_state_dict(model)

# model_eff =nn.Sequential(*list(models_eff.children())[:-1])

# model_eff.load_state_dict(model["model_state_dict"])
# print(model_eff)
# 

# model = build_model()

models_eff = models_eff.cuda()
# # model.eval()

image = cv2.imread("/home/iw/Downloads/scripts/temp.jpg")
image2 = cv2.imread("/home/iw/Downloads/scripts/temp.jpg")

# # # image = Image.open("/home/sr/Pictures/infrence_image.png")
# # batch_size = 1

# # # image = cv2.resize(image, (64,64))

image = cv2.resize(image,(224,224))
image2 = cv2.resize(image2,(224,224))


print(image.shape)

# # # image  = image.astype(float)

from torchvision import transforms
tr = transforms.ToTensor()

image_tensor = tr(image)
image_tensor2 = tr(image2)


print(image_tensor.shape)

array_image = [image_tensor, image_tensor2]


# print(len(array_image))

images = torch.stack(array_image)

print("THi is shape ",images.shape)


image_tensor = image_tensor.unsqueeze(0).cuda()

print("Model output is the ",models_eff(image_tensor))


print(image_tensor.shape)


input_names = ["input"]
output_names = ["output"]

torch.onnx.export(models_eff,
    image_tensor,
    "Wagon.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    export_params=True,
    opset_version=11,
    do_constant_folding = True,
    dynamic_axes={'input' : {0 : 'batch_size'},   
                                'output' : {0 : 'batch_size'}})

# print(model)


apply(change_input_dim, "Wagon.onnx","Wagonb.onnx")

with torch.no_grad():
    ort = onnxruntime.InferenceSession("Wagonb.onnx")
    ort_inp = {ort.get_inputs()[0].name:to_numpy(images)}
    ort_out = ort.run(None, ort_inp)

    print("THis is the tensorrt output las")

    print(ort_out[0])
