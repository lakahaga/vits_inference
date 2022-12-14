chmod u+x *.sh
pip install -r requirements.txt
./install_cuda_compatible_torch.sh
python export_onnx.py
python infer.py
./prepare_dir.sh
./ngc_login.sh
./install_triton.sh
