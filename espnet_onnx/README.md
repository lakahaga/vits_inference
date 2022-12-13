# [espnet_onnx](https://github.com/espnet/espnet_onnx) 에서 수정한 사항

1. `espnet_onnx/export/tts/get_config.py`
   * Text Cleaner가 빈 리스트로 설정 되어 있을 때 None으로 설정하도록 수정
   * line 45 `get_preprocess_config`
     ```python
         if model.text_cleaner is not None and len(model.text_cleaner.cleaner_types) > 0:
        ret.update({'text_cleaner': {
                'cleaner_types': model.text_cleaner.cleaner_types[0]
        }})
   
2. `espnet_onnx/export/tts/export_tts.py` 
   * File Not Found Error 나는 부분 수정
     * line 57
       ```python
       base_dir.mkdir(parents=True, exist_ok=True)

     * line 207
        * `{model_name}.onnx` 로 저장되는데 `{model_name}-opt.onnx` 를 찾아서 모델명 수정
       ```python
        os.remove(os.path.join(model_from, basename + '.onnx'))