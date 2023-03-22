# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 16:03
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : torch_2_onnx.py
# @Software: PyCharm
import onnx
import torch
from mmseg.utils import register_all_modules
register_all_modules()
from mmseg.apis import init_model, inference_model
from onnxsim import simplify
import sys
sys.path.extend(['C:\\Users\\liuye\\Desktop\\workspace\\liuye\\5_Remote_Image_CloudSeg'])
import runner

ONNX_MODEL_PATH = "CloudSeg.onnx"
ONNX_SIM_MODEL_PATH = "CloudSeg_SIM.onnx"
model = init_model('./custom_config/ConvNext/ConvNext_tiny.py',
                   checkpoint='../3_results/5_Remote_Image_CloudSeg/best_mIoU_epoch_37.pth'
                   )

dummy_input = torch.randn(1, 3, 256, 256, device='cuda')
torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, verbose=True, opset_version=11,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
onnx_sim_model, check = simplify(onnx.load(ONNX_MODEL_PATH))
assert check, "Simplified ONNX model could not be validated"
onnx.save(onnx_sim_model, ONNX_SIM_MODEL_PATH)
print('ONNX file simplified!')
