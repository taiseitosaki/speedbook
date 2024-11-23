import torch
from torchvision.models import resnet50, ResNet50_Weights

# ResNet50の準備
profile_model = resnet50(weights=ResNet50_Weights.DEFAULT)
profile_model = profile_model.to('cuda')
profile_model.eval()

# ダミー画像を用意
input_image = torch.ones((512, 3, 224, 224))
input_image = input_image.to('cuda')

with torch.autograd.profiler.emit_nvtx():
    with torch.no_grad():
        output = profile_model(input_image)