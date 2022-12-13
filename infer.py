import re
import soundfile as sf
import kaldiio
import json

import onnx
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
            self.speaker_table = kaldiio.load_ark(speaker_embedding)
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


    def infer(self, input, out_file, **kwargs):
        input = self._text_processing(input)
        input_ids = self.tts_model.preprocess_fn("<dummy>", dict(text=input))["text"]

        if self.tts_model.use_spembs:
            if "speaker" not in kwargs.keys():
                raise ValueError("This TTS model uses speaker embedding. Speaker should be given. Possible spaekers can be found in speaker_info.json")
            spk_emb = self._get_speaker_embedding(kwargs['speaker'])
            wav = self.tts_model(input_ids, spembs=spk_emb)['wav']
        else:
            wav = self.tts_model(input_ids)['wav']

        sf.write(out_file, wav, samplerate=self.sample_rate, format='WAV', subtype="PCM_16")

class OnnxInfer(object):
    pass

if __name__ =="__main__":

    EspnetInfer(checkpoint_path="vits_model/273epoch.pth")