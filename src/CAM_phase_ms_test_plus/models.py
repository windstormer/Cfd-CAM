import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from collections import OrderedDict
from torchvision.models.segmentation import deeplabv3_resnet50

class Res18_Classifier(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res18_Classifier, self).__init__()

        resnet = resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.decoder = nn.Sequential(
            nn.Linear(512, 1)
            )
            
    def forward(self, x):
        mask = self.f(x)
        x = self.gap(mask)
        feature = torch.flatten(x, start_dim=1)
        pred = self.decoder(feature)
        return mask, feature, pred

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")

class Res50_Classifier(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res50_Classifier, self).__init__()

        resnet = resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1)
            )
            
    def forward(self, x):
        mask = self.f(x)
        x = self.gap(mask)
        feature = torch.flatten(x, start_dim=1)
        pred = self.decoder(feature)
        return mask, feature, pred

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")