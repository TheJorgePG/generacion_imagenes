import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from huggingface_hub import login, HfApi
from tqdm.auto import tqdm

def main():
    # 1. Configuración inicial
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    DATASET_NAME = "gigant/oldbookillustrations"
    OUTPUT_DIR = "./oldbooks_lora"
    
    print(f"--> Usando dispositivo: {device}")

    # 2. Cargar el dataset de ilustraciones antiguas
    print("--> Cargando dataset de Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train")

    dataset = dataset.select(range(500))

    # 3. Transformaciones para imágenes con alturas variables
    image_resolution = 512
    train_transforms = transforms.Compose([
        transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_batch(examples):
        images = [train_transforms(img.convert("RGB")) for img in examples["1600px"]]
        texts = examples["info_alt"]
        return {"pixel_values": images, "caption": texts}

    dataset.set_transform(transform_batch)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Cargar componentes del modelo base
    print("--> Cargando componentes de Stable Diffusion...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", use_fast=False)
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    
    # Cargar pipeline temporal solo para extraer el UNet original
    pipeline_temp = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    unet = pipeline_temp.unet
    del pipeline_temp

    # Congelar componentes que no se entrenan
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # 5. Configurar LoRA
    print("--> Configurando LoRA para el UNet...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 6. Optimizador y planificador de tasa de aprendizaje
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * 1  
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    # 7. Bucle de entrenamiento
    print("--> Iniciando el entrenamiento...")
    unet.train()
    epochs = 1

    for epoch in range(epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{epochs}")
        for step, batch in enumerate(progress_bar):
            # Codificar imágenes a latentes con el VAE
            pixel_values = batch["pixel_values"].to(device)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Añadir ruido aleatorio
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenizar las descripciones (info_alt)
            inputs = tokenizer(
                batch["caption"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            # Predicción del ruido por parte del UNet
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calcular pérdida (Loss)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

    # 8. Guardar el modelo LoRA entrenado localmente
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR)
    print(f"--> ¡Entrenamiento finalizado! Pesos guardados en '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()