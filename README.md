Fine-Tuning LoRA: Stable Diffusion v1-4 + Old Book Illustrations

Este repositorio contiene el código y los resultados del proceso de fine-tuning realizado sobre el modelo Stable Diffusion v1-4 utilizando la técnica de adaptación de bajo rango (LoRA).

El objetivo del proyecto es adaptar el modelo base para que genere imágenes con el estilo característico de los grabados e ilustraciones de libros antiguos.

Estructura del Repositorio

train_lora.py: Script principal utilizado para el entrenamiento del adaptador LoRA.

generar_comparativa.py: Script de inferencia para generar imágenes antes y después del fine-tuning bajo el mismo prompt y semilla.

link_huggingface.txt: Enlace directo al repositorio de Hugging Face donde se encuentran almacenados los pesos (adapter_model.safetensors) y la configuración (adapter_config.json).

imagen_antes.png: Resultado generado por el modelo base de Stable Diffusion v1-4 sin adaptar.

imagen_despues.png: Resultado obtenido aplicando los pesos LoRA entrenados.

Datos y Configuración de Entrenamiento

Modelo base: CompVis/stable-diffusion-v1-4

Dataset de entrenamiento: gigant/oldbookillustrations

Método de ajuste: LoRA (peft) aplicado al módulo UNet del pipeline.

Muestras procesadas: Subconjunto de 500 imágenes extraídas del dataset original para la adaptación de dominio.

Comparativa de Resultados

Prompt utilizado:

"An old engraving of a botanical flower illustration, vintage style"

Pesos del Modelo

Los pesos finales del adaptador LoRA no están subidos directamente a Git por limitaciones de espacio, pero se pueden consultar y descargar desde el repositorio público de Hugging Face enlazado en link_huggingface.txt.