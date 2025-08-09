# from model import TransformerNet
# from torchvision.transforms import v2
# import torch
# import onnxruntime
#
#
# class InferenceProcess:
#     def __init__(self, model_path: str) -> None:
#         self.ort_session = onnxruntime.InferenceSession(
#             model_path,
#             providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
#         )
#         self.model = TransformerNet()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     @staticmethod
#     def _preprocess(image):
#         transform = v2.Compose([
#             v2.ToImage(),
#             v2.Resize((1080, 1080)),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 255.0, 1 / 255.0, 1 / 255.0])
#         ])
#
#         # Apply transformations
#         image_tensor = transform(image)
#         # Add batch dimension
#         image_batch = image_tensor.unsqueeze(0)
#
#         return image_batch
#
#     def __call__(self, image):
#         preprocessed_input = self._preprocess(image)
#         preprocessed_input = preprocessed_input.to(self.device)
#
#         def to_numpy(tensor):
#             return (
#                 tensor.detach().cpu().numpy()
#                 if tensor.requires_grad
#                 else tensor.cpu().numpy()
#             )
#
#         ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(preprocessed_input)}
#         ort_outs = self.ort_session.run(None, ort_inputs)
#         img_out_y = ort_outs[0]
#
#         output = torch.from_numpy(img_out_y)
#
#         return output
from model import TransformerNet
from torchvision.transforms import v2
import torch
import onnxruntime
from PIL import Image

class InferenceProcess:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.model = TransformerNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        # 从全局定义的 MODEL_SIZES 获取尺寸（需确保 app.py 已定义）
        from app import MODEL_SIZES  # 动态导入
        self.expected_height, self.expected_width = MODEL_SIZES.get(model_name, (1080, 1080))

    @staticmethod
    def _preprocess(image, target_height, target_width):
        # 强制调整到目标尺寸，忽略宽高比
        img = image.convert('RGB')
        img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        ])

        image_tensor = transform(img_resized)
        image_batch = image_tensor.unsqueeze(0)

        return image_batch

    def __call__(self, image):
        preprocessed_input = self._preprocess(image, self.expected_height, self.expected_width)
        preprocessed_input = preprocessed_input.to(self.device)

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(preprocessed_input)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        img_out_y = ort_outs[0]

        output = torch.from_numpy(img_out_y)

        return output