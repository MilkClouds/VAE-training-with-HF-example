# accelerate launch --multi_gpu train.py
# train VAE with mnist, use huggingface libraries(datasets, accelerate, diffusers, ...)
from pathlib import Path
from types import SimpleNamespace
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import albumentations as A
import wandb
from albumentations.pytorch.transforms import ToTensorV2
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import AutoencoderKL

transform_train = A.Compose([
    A.Resize(32, 32), # ensure image size is 32x32
    # A.HorizontalFlip(),
    A.ToFloat(255),
    ToTensorV2()
])
transform_valid = A.Compose([
    A.Resize(32, 32),
    A.ToFloat(255),
    ToTensorV2()
])

def transforms(examples):
    examples["image"] = [transform_train(image=np.array(image.convert("RGB")))['image'] for image in examples["image"]]
    return examples

def vae_loss(vae, data):
    out = vae.encode(data)
    KLE = torch.sum(out.latent_dist.kl())
    out = vae.decode(out.latent_dist.sample())
    # calculate MSE loss
    B, C, H, W = data.shape
    # caution:
    # torch.mean(KLE) != mse_loss(reduction='mean')
    # torch.mean(KLE) == mse_loss(reduction='mean') * (C*H*W) == mse_loss(reduction='sum') / B
    MSE = F.mse_loss(data, out.sample, reduction='sum')
    MSE /= B
    KLE /= B
    return MSE, KLE

def prepare(config):
    model = AutoencoderKL(
        in_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64, 128),
        layers_per_block=2,
    )
    # for non-"load_dataset" method such that do not consider multinode training,
    # you should check whether the dataset is downloaded on main process
    # it can be checked with `accelerator.is_main_process`
    # https://huggingface.co/docs/accelerate/v0.20.3/en/concept_guides/deferring_execution
    dataset = load_dataset('mnist')
    dataset.set_transform(transforms)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
    criterion = config.criterion
    return model, optimizer, scheduler, dataset, criterion

def train(config):
    model, optimizer, scheduler, dataset, criterion = prepare(config)

    # count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(model)

    train_dataset, valid_dataset = dataset['train'], dataset['test']
    # for multi-gpu training, it's best practice to:
    # 1. split batch size by number of gpus, OR
    # 2. multiply learning rates by number of gpus
    # https://huggingface.co/docs/accelerate/v0.20.3/en/concept_guides/performance
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = False)

    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers("VAE", config=config)
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, valid_loader)
    for epoch in range(1, 1 + config.num_epochs):
        # train
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            image = batch['image']
            MSE, KLE = criterion(model, image) # vae loss = reconstruction loss + KLE
            loss = MSE + KLE * config.kl_weight
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accelerator.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_loss": loss.item(), "train_MSE": MSE.item(), "train_KLE": KLE.item()})
        # valid
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader)):
                image = batch['image']
                label = batch['label']
                MSE, KLE = criterion(model, image) # vae loss = reconstruction loss + KLE
                loss = MSE + KLE * config.kl_weight
                accelerator.log({"epoch": epoch, "val_loss": loss.item(), "val_MSE": MSE.item(), "val_KLE": KLE.item()})
                # log 3 reconstructed image
                log_images = image[:3]
                recon_images = model(log_images).sample
                z = torch.randn(3, 4, 32 // 2 ** 3, 32 // 2 ** 3).to(accelerator.device)
                sample_images = model.decode(z).sample
                # ==============
                if idx == 0:
                    accelerator.log({"epoch": epoch,
                                    "original_images": [wandb.Image(image) for image in log_images],
                                    "reconstructed_images": [wandb.Image(image) for image in recon_images],
                                    "sample_images": [wandb.Image(image) for image in sample_images],
                                    "label": label[:3]})
        accelerator.wait_for_everyone()
        accelerator.save_state(f"outputs/{epoch}")
    accelerator.end_training()


def sample(config):
    model, optimizer, scheduler, dataset, criterion = prepare(config)

    valid_dataset = dataset['test']
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = False)

    accelerator = Accelerator()
    accelerator.init_trackers("VAE", config=config)
    model, optimizer, scheduler, valid_loader = accelerator.prepare(model, optimizer, scheduler, valid_loader)
    accelerator.load_state(f"outputs/{config.num_epochs}")
    model.eval()
    with torch.no_grad():
        z = torch.randn(3, 4, 32 // 2 ** 3, 32 // 2 ** 3).to(accelerator.device)
        sample_images = model.decode(z).sample
        # save img locally
        for idx, image in enumerate(sample_images):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"outputs/sample_{idx}.png")
        for idx, batch in enumerate(tqdm(valid_loader)):
            image = batch['image']
            label = batch['label']
            MSE, KLE = criterion(model, image) # vae loss = reconstruction loss + KLE
            loss = MSE + KLE * config.kl_weight
            latent_dist = model.encode(image).latent_dist
            print(latent_dist.mean.mean())
            print(latent_dist.std.mean())

def visualize(config):
    model, optimizer, scheduler, dataset, criterion = prepare(config)

    valid_dataset = dataset['test']
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = False)
    
    accelerator = Accelerator()
    accelerator.init_trackers("VAE", config=config)
    model, optimizer, scheduler, valid_loader = accelerator.prepare(model, optimizer, scheduler, valid_loader)
    accelerator.load_state(f"outputs/{config.num_epochs}")
    model.eval()
    data = {}
    if Path("outputs/latent_space.npy").exists():
        with open("outputs/latent_space.npy", "rb") as f:
            data = np.load(f, allow_pickle=True).item()
    else:
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader)):
                image = batch['image']
                label = batch['label']
                MSE, KLE = criterion(model, image) # vae loss = reconstruction loss + KLE
                loss = MSE + KLE * config.kl_weight
                latent_dist = model.encode(image).latent_dist
                for idx, (mean, std) in enumerate(zip(latent_dist.mean, latent_dist.std)):
                    l = label[idx].item()
                    data[l] = np.concatenate((data.get(l, np.zeros((0, 256))), [mean.detach().cpu().numpy().reshape(-1)]), axis=0)
        with open("outputs/latent_space.npy", "wb") as f:
            np.save(f, data)
    # visualize data with two dimension figure
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    labels = []
    means = []
    for label, mean in data.items():
        labels.extend([label]*mean.shape[0])
        means.extend(mean)
    labels = np.array(labels)
    means = np.array(means)
    print(labels.shape)
    print(means.shape)
    if Path("outputs/tsne.npy").exists():
        with open("outputs/tsne.npy", "rb") as f:
            means = np.load(f)
    else:
        means = tsne.fit_transform(means)
        with open("outputs/tsne.npy", "wb") as f:
            np.save(f, means)
    print(means.shape, labels.shape)
    # data = {label: (tsne.fit_transform(mean.reshape(mean.shape[0], -1)), std) for label, (mean, std) in data.items()}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(means[idx, 0], means[idx, 1], label=label)
    plt.legend()
    plt.savefig("outputs/latent_space.png")

if __name__ == '__main__':
    config = SimpleNamespace()
    config.kl_weight = 1
    config.criterion = vae_loss
    config.batch_size = 768
    config.lr = 1e-3
    config.num_epochs = 10
    train(config)
    sample(config)
    visualize(config)