import torch
import torch.nn.functional as F

def get_cam_pred(model, xs):
    features = model._features(xs)
    model_fc = model._classifier
    cam = F.conv2d(features, model_fc.weight.view(model_fc.out_features, features.size(1), 1, 1)) + model_fc.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return cam
