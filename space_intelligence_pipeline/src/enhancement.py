"""
Image Enhancement module using Real-ESRGAN super-resolution.
Sharpens low-quality/compressed space imagery (lunar, Mars, satellite photos).
"""
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from basicsr.archs.rrdbnet_arch import RRDBNet
# pyrefly: ignore [missing-import]
from realesrgan import RealESRGANer
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import cv2


class ImageEnhancer:
    def __init__(self, model_path: str = "weights/RealESRGAN_x4plus.pth"):
        self.gpu_available = torch.cuda.is_available()

        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=self.model,
            tile=0 if self.gpu_available else 200,   # no tiling needed on GPU for small images
            tile_pad=10,
            pre_pad=0,
            half=self.gpu_available                   # fp16 only safe/fast on GPU
        )

    def enhance(self, image_path: str, output_path: str, scale: int = 4,
                max_input_size: int = 800) -> str:
        """
        Enhances a space image using Real-ESRGAN.

        Args:
            image_path: path to input image
            output_path: path to save enhanced image
            scale: upscaling factor (default 4x)
            max_input_size: resize input down if larger than this, to avoid
                             out-of-memory crashes on free-tier Colab

        Returns:
            output_path: path to the saved enhanced image
        """
        img_pil = Image.open(image_path).convert('RGB')
        w, h = img_pil.size
        if max(w, h) > max_input_size:
            ratio = max_input_size / max(w, h)
            img_pil = img_pil.resize((int(w * ratio), int(h * ratio)),
                                      Image.Resampling.LANCZOS)
            img_pil.save('resized_input.jpg')
            image_path = 'resized_input.jpg'

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        output, _ = self.upsampler.enhance(img, outscale=scale)
        cv2.imwrite(output_path, output)
        return output_path

