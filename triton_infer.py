import numpy as np
import soundfile as sf

from infer import TritionInfer

input_ids = np.load("input_ids.npy")
speaker_embedding = np.load("speaker_embedding.npy")


runner = TritionInfer(url='127.0.0.1:8000', model_name='korean_vits', model_version='1', verbose=False)
wav = runner.infer(input_ids, speaker_embedding)
sf.write("triton_output.wav", wav, samplerate=22050, format='WAV', subtype='PCM_16')