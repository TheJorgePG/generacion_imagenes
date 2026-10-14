import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# Configuración del dispositivo y un prompt representativo de ilustraciones antiguas
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "An old engraving of a botanical flower illustration, vintage style"

print("--> Generando imagen ANTES (Modelo base)...")
# 1. Cargar el modelo base original de Stable Diffusion v1-4
pipe_base = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

image_base = pipe_base(prompt, num_inference_steps=30).images[0]
image_base.save("imagen_antes.png")
print("--> Guardada: imagen_antes.png")

# Liberar memoria RAM/VRAM del modelo base
del pipe_base
torch.cuda.empty_cache()

print("--> Generando imagen DESPUÉS")
# 2. Cargar el pipeline base y aplicarle los pesos LoRA entrenados
pipe_lora = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Cargar los pesos que se guardaron en la carpeta del entrenamiento
pipe_lora.unet = PeftModel.from_pretrained(pipe_lora.unet, "./oldbooks_lora")

image_lora = pipe_lora(prompt, num_inference_steps=30).images[0]
image_lora.save("imagen_despues.png")
print("--> Guardada: imagen_despues.png")

print("¡Proceso de comparación finalizado con éxito!")