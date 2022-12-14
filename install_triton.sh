docker pull nvcr.io/nvidia/tritonserver:22.11-py3
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/erin/inference/onnx_model:/models "nvcr.io/nvidia/tritonserver:22.11-py3" tritonserver --model-repository=/models
curl -v localhost:8000/v2/health/ready