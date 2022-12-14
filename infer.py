import time

import re
import soundfile as sf
import kaldiio
import json

import torch
import numpy as np

import onnxruntime
from espnet2.bin.tts_inference import Text2Speech


class EspnetInfer(object):
    def __init__(self, checkpoint_path=None, model=None, speaker_embedding=None, speaker_info=None):
        if model is None:
            if checkpoint_path is None:
                raise ValueError("Checkpoint path should be specified when model is not given")
            model = Text2Speech.from_pretrained(model_file=checkpoint_path)
        self.tts_model = model
        self.tokenizer = model.preprocess_fn.tokenizer
        self.sample_rate = model.train_args.tts_conf['sampling_rate']

        if self.tts_model.use_spembs:
            if speaker_embedding is None:
                raise ValueError("This TTS model uses speaker embedding. You should specify speaker embedding")
            self.speaker_table = {}
            for key, array in kaldiio.load_ark(speaker_embedding):
                self.speaker_table[key] = array
            if speaker_info is not None:
                self.speaker_info = json.load(open(speaker_info))

    def _text_processing(self,text):
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
        tokens = self.tokenizer.text2tokens(text)
        token_list = []
        for token in tokens:
            if token == ' ':
                continue
            token_list.append(token)
        return " ".join(token_list)

    def _get_speaker_embedding(self, speaker):
        return self.speaker_table[speaker]

    def get_input_ids(self, text):
        input = self._text_processing(text)
        input_ids = self.tts_model.preprocess_fn("<dummy>", dict(text=input))["text"]
        return input_ids


    def infer(self, input, out_file, **kwargs):
        input = self._text_processing(input)
        input_ids = self.tts_model.preprocess_fn("<dummy>", dict(text=input))["text"]

        if self.tts_model.use_spembs:
            if "speaker" not in kwargs.keys():
                raise ValueError("This TTS model uses speaker embedding. Speaker should be given. Possible spaekers can be found in speaker_info.json")
            spk_emb = self._get_speaker_embedding(kwargs['speaker'])
            start = time.perf_counter()
            wav = self.tts_model(input_ids, spembs=spk_emb)['wav']
            infer_time = time.perf_counter() - start
        else:
            start = time.perf_counter()
            wav = self.tts_model(input_ids)['wav']
            infer_time = time.perf_counter() - start
        print(f"Espnet Inference time: {infer_time}")
        sf.write(out_file, wav, samplerate=self.sample_rate, format='WAV', subtype="PCM_16")
        return wav

class OnnxInfer(object):
    def __init__(self, model: str):
        self.session = onnxruntime.InferenceSession(model)

    def to_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
        return x

    def _get_input(self,input_ids, speaker_emb):
        return {
            self.session.get_inputs()[0].name : self.to_numpy(input_ids),
            self.session.get_inputs()[1].name : self.to_numpy(speaker_emb)
        }

    def infer(self, input_ids,spk_emb):
        input = self._get_input(input_ids, spk_emb)
        start = time.perf_counter()
        output = self.session.run(None, input)
        infer_time = time.perf_counter() - start
        print(f"Onnx Inference time: {infer_time}")
        return output[0]

    def compare_with_torch(self, onnx_output, torch_output):
        if torch.is_tensor(onnx_output):
            onnx_output = onnx_output.detach().cpu().numpy() if onnx_output.requires_grad else onnx_output.cpu().numpy()
        if torch.is_tensor(torch_output):
            torch_output = torch_output.detach().cpu().numpy() if torch_output.requires_grad else torch_output.cpu().numpy()

        try:
            np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-05)
        except AssertionError:
            return False, f"Not equal to tolerance rtol={0.001}, atol={1e-05}"
        return True, "Onnx output is equal to torch output with tolerance rtol={0.001}, atol={1e-05}"

if __name__ =="__main__":

    runner = EspnetInfer(checkpoint_path="vits_model/273epoch.pth", speaker_embedding='vits_model/spk_xvector.ark')
    kwargs = {'speaker' : "s101_annie_angry"}
    torch_output = runner.infer("안녕하세요", 'sample.wav', **kwargs)

    input_ids = runner.get_input_ids("안녕하세요")
    np.save("input_ids.npy", input_ids)
    speaker_embedding = runner._get_speaker_embedding("s101_annie_angry")
    np.save("speaker_embedding.npy", speaker_embedding)

    torch_output2 = runner.infer("안녕하세요", 'sample.wav', **kwargs)

    # onnx_runner = OnnxInfer(model="export_dir/korean_vits/quantize/vits_qt.onnx")
    onnx_runner = OnnxInfer(model="export_dir/korean_vits/full/vits.onnx")
    onnx_output = onnx_runner.infer(input_ids, speaker_embedding)
    sf.write("sample_onnx.wav", onnx_output, samplerate=22050,format='WAV', subtype="PCM_16")
    is_valid, msg = onnx_runner.compare_with_torch(onnx_output, torch_output)
    print(msg)

    print("Compare two torch output")
    is_valid, msg = onnx_runner.compare_with_torch(torch_output, torch_output2)
    print(msg)