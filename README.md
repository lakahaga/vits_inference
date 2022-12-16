# vits_inference
한국어 VITS 모델 optimize 및 inference server porting

## How to run
* You can use `run.sh` to run the whole project from environment setup to model exporting.
* You should enter your ngc api key to `ngc_login.sh` before download triton server image.

## Inference Environment
* `GPU` : V100 32GB
* `CUDA` : 11.3

## Result
```shell
Espnet Inference time: 0.3507644236087799
Onnx Inference time: 0.09777828399091959
Triton Inference time : 0.33419392816722393
```

## Exported Models
* You can download models [here](https://sogang365-my.sharepoint.com/:f:/g/personal/lakahaga_o365_sogang_ac_kr/ErIEGeXHq4ZMk1fZICilGgcB1YgWA4k3ArUy_rIv_WL91Q?e=pETg5i)
* `model_vits` : original VITS model trained with Espnet
* `export_dir` : onnx exported models
* `onnx_model` : directory for triton server
