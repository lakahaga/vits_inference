# vits_inference
한국어 VITS 모델 optimize 및 inference server porting

## Result
```shell
Espnet Inference time: 0.3507644236087799
Onnx Inference time: 0.09777828399091959
Triton Inference time : 0.33419392816722393
```

## Exported Models
* You can download models [here](https://drive.google.com/drive/folders/1VUvmA3K3T8QuW1RkGWSz2xin2j1ZefSq?usp=share_link)
* `model_vits` : original VITS model trained with Espnet
* `export_dir` : onnx exported models
* `onnx_model` : directory for triton server
