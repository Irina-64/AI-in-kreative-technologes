# Журнал измерений для лекции 4 (Google Colab)

Дата: 19.09.2026. Блокнот `lec04.ipynb` (Google Диск проверявшего), создан отдельно от `lab05.ipynb` и `lab06.ipynb`. Ускоритель Tesla T4 (первая попытка на CPU завершилась ошибкой «Torch not compiled with CUDA enabled»; ускоритель переключён на T4, среда пересоздана). Python 3.13.15, torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1. Библиотеки `lab05_lib.py` и `lab06_lib.py` введены с проверкой SHA-256 (`4eb17650…`, `c6d8473c…`).

```text
I scheduler: EulerDiscreteScheduler timestep_spacing: trailing
I text_encoder: CLIPTextModel max tokens: 77
I unet: UNet2DConditionModel params, M: 865.9
I text_encoder params, M: 340.4
I vae params, M: 83.7 latent channels: 4 vae_scale_factor: 8
ST 1 clip 30.7956 s1.3941 edge 0.0457 s0.0069 contrast 0.1923 s0.0226 seconds 0.2836 s0.0023
ST 2 clip 30.5507 s2.1477 edge 0.0778 s0.0088 contrast 0.2470 s0.0111 seconds 0.3648 s0.0007
ST 3 clip 30.1664 s1.8810 edge 0.0888 s0.0084 contrast 0.2699 s0.0123 seconds 0.4461 s0.0013
ST 4 clip 30.2919 s2.1490 edge 0.0920 s0.0085 contrast 0.2778 s0.0192 seconds 0.5438 s0.0057
NEG guidance 0.0 with and without negative prompt: SHA equal True
NEG guidance 3.0 with and without negative prompt: SHA equal False PSNR 11.54
SEEDS pairwise PSNR min/mean/max: 9.55 10.13 10.74   (4 seed, бриф 1)
```
