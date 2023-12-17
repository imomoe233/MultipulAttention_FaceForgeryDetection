import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys


sys.path.append(".")
sys.path.append("..")

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = TRANSFORM(image).unsqueeze(0)
    return image.to(device)


def load_parameter(param_path, device):
    parameter = torch.zeros([1, 25], device=device)
    parameter_np = np.load(param_path)
    for i in range(parameter_np.__len__()):
        parameter[0, i] += parameter_np[i]
    return parameter


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def configure_optimizers(networks, lr=3e-4):
    params = list(networks.backbone.parameters()) + list(networks.renderer.parameters()) + list(networks.decoder.parameters())
    optimizer = torch.optim.Adam([{'params': params}], lr=lr)
    return optimizer


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size

