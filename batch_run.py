import os
import subprocess

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
models = ['WavePCNet']
gpus = '0'
trset = 'HKU_train'
valset = 'HKU_test'
batch_size = 3
weight_base_path = './weight/'
train_script = 'train.py'
test_script = 'test.py'

for model in models:
    print(f"\n================ {model} ================\n")

    train_cmd = f"python3 {train_script} {model} --gpus={gpus} --trset={trset} --batch={batch_size} --val={valset} --resume --weight=./weight/ECSSD/"
    ret_train = subprocess.call(train_cmd, shell=True)
    
    if ret_train != 0:
        print(f"[!] ：{model}")
        continue

    model_dir = os.path.join(weight_base_path, model, 'resnet50', 'base')
    weight_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith(f"{model}_resnet50")]
    if not weight_files:
        print(f"[!] can't find  {model} ")
        continue

    weight_files.sort(key=lambda x: int(x.split('_')[-2][5:]))
    latest_weight = weight_files[-1]
    weight_path = os.path.join(model_dir, latest_weight)

    print(f"\n>>> : {model}")
    print(f"    : {weight_path}\n")

    test_cmd = f"python3 {test_script} {model} --gpus={gpus} --weight={weight_path} --save --val={valset}"
    ret_test = subprocess.call(test_cmd, shell=True)

    if ret_test != 0:
        print(f"[!] ：{model}")
    else:
        print(f"[✓] ：{model}\n")

print("\n================================\n")
