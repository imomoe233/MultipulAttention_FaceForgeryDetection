import torch
from models.encoders.psp_encoders import GradualStyleEncoder
import torch.nn.functional as F

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class InversionCoach:
    def __init__(self):
        self.device = torch.device('cuda')
        self.encoder = self.load_encoder()

    def load_encoder(self):
        encoder = GradualStyleEncoder(50, 'ir_se')
        encoder_ckpt = torch.load('checkpoints/encoder.pt')
        encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
        return encoder

    def run(self, image):
        x = F.interpolate(image, size=[256, 256], mode='bilinear', align_corners=True)
        with torch.no_grad():
            latent_code = self.encoder(x.cpu()).to(self.device)
            
        return latent_code
