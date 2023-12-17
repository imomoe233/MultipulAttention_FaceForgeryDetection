import torch
from models.encoders.psp_encoders import GradualStyleEncoder
import torchvision.transforms as transforms
from PIL import Image

class InversionCoach:
    def __init__(self, encoder_checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self._load_encoder(encoder_checkpoint_path)
        self.transform = self._create_transform()

    def _load_encoder(self, checkpoint_path):
        # 加载编码器模型
        encoder = GradualStyleEncoder(50, 'ir_se').to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        encoder.load_state_dict(checkpoint, strict=True)
        encoder.eval()  # 设置为评估模式
        return encoder

    def _create_transform(self):
        # 图像预处理，调整大小并标准化
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 或者根据您的模型更改为其他尺寸
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform

    def encode_image(self, image_path):
        # 从文件中加载图像，应用转换并进行编码
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)  # 添加批处理维度并转移到设备
        with torch.no_grad():
            encoding = self.encoder(image)
        return encoding  # 或者任何您需要的后处理


# 使用方法示例：
# encoder_path = 'path_to_your_encoder_checkpoint/encoder.pt'
# inverter = SimpleInverter(encoder_checkpoint_path=encoder_path)
# image_path = 'path_to_your_image.jpg'
# encoded_image = inverter.encode_image(image_path)
# # 现在，encoded_image包含图像的编码结果
