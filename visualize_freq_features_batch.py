import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import scipy.io
from base.framework_factory import load_framework

from torchvision import transforms
from PIL import Image
import argparse
def load_png_image(png_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(png_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor

def fft_features(feature_map):
    f = np.fft.fft2(feature_map)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    pha = np.angle(fshift)
    return np.log1p(mag), pha

def visualize_feature(original, magnitude, phase, save_path, layer_name, image_name):
    os.makedirs(save_path, exist_ok=True)

    plt.figure()
    plt.imshow(original, cmap='viridis')
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f"{image_name}_{layer_name}_original.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(magnitude, cmap='hot')
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f"{image_name}_{layer_name}_magnitude.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(phase, cmap='twilight_shifted')
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f"{image_name}_{layer_name}_phase.png"), bbox_inches='tight', pad_inches=0)
    plt.close()



def get_module_by_name(model, name):
    parts = name.split('.')
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod

def extract_and_visualize(model, image_tensor, target_layers, save_dir, image_name):
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    hooks = [get_module_by_name(model, name).register_forward_hook(hook_fn(name)) for name in target_layers]
    _ = model(image_tensor)

    for name, feat in features.items():
        fmap = feat[0].mean(0).cpu().numpy()
        mag, pha = fft_features(fmap)
        visualize_feature(fmap, mag, pha, save_dir, name.replace('.', '_'), image_name)

    for h in hooks:
        h.remove()

def build_model(weight_path, device):
    config, model, _, _, _, _ = load_framework('meanetA_3')
    config['device'] = device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    image_dir = ''
    weight_path = ''
    save_dir = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(weight_path, device)

    target_layers = [
        "S1.1",
        "S2.1",
        "S3.1",
        "S4.1",
        "S5.1",
    ]

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    for fname in files:
        try:
            fpath = os.path.join(image_dir, fname)
            image = load_png_image(fpath).to(device)

            extract_and_visualize(
                model,
                image,
                target_layers,
                save_dir,
                os.path.splitext(fname)[0]
            )

            print(f"✅ ：{fname}")
        except Exception as e:
            print(f"❌ {fname}，reason：{e}")
