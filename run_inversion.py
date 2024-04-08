import os
from models.inversion_coach import InversionCoach


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

coach = InversionCoach('encoder_checkpoint_path')
coach.encode_image('image_path')
