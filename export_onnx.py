import onnx
from espnet2.bin.tts_inference import Text2Speech
import sys
sys.path.append("espnet_onnx")
from espnet_onnx.export import TTSModelExport

ckp_path = "vits_model/273epoch.pth"
text2speech = Text2Speech.from_pretrained(model_file=ckp_path)
m = TTSModelExport(cache_dir='export_dir')
tag_name = 'korean_vits'
m.export(text2speech, tag_name, quantize=True, verbose=True)

# onnx check
onnx_model= onnx.load("export_dir/korean_vits/quantize/vits_qt.onnx")
onnx.checker.check_model(onnx_model)
