import torch
import torch.nn as nn
import torchvision
import os



def get_model(model_name: str, num_classes: int, weights:str, checkpoint_path: str = None):

    if model_name not in ['resnet18', 'resnet34', 'resnet50',
                          'densenet121', 'mobilenet_v2', 'shufflenet_v2']:
        raise ValueError(f"Model {model_name} not supported")
    
    if weights not in ['default', 'none']:
        raise ValueError(f"Weights {weights} not supported")

    print(f'loading {model_name} model with {weights} weights')
    if model_name == 'resnet18':
        weights = torchvision.models.ResNet18_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'resnet34':
        weights=torchvision.models.ResNet34_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.resnet34(weights=weights)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'densenet121':
        weights=torchvision.models.DenseNet121_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.densenet121(weights=weights)
        model.classifier = nn.Linear(1024, num_classes)
    elif model_name == 'mobilenet_v2':
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    elif model_name == 'shufflenet_v2':
        weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT if weights == 'default' else None
        model = torchvision.models.shufflenet_v2_x0_5(weights=weights)
        model.fc = nn.Linear(1024, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Check if the checkpoint is a state_dict or contains a state_dict under a key (common in training scripts)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif checkpoint_path is not None:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return model