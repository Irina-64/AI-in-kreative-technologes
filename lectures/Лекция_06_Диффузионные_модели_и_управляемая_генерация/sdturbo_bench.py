"""Замер скорости SD Turbo на процессоре Engee (лекция 6, ЛР 9-10).
Запуск: python sdturbo_bench.py <потоки>   Вывод: строки JSON в bench_<потоки>.jsonl
"""
import json, resource, sys, time
import torch

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.set_num_threads(THREADS)
MODEL = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
PROMPT = "a photo of a red cup on a wooden table, soft daylight"
OUT = f"bench_{THREADS}.jsonl"


def log(**kw):
    kw["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")


import diffusers, transformers
log(stage="versions", torch=torch.__version__, diffusers=diffusers.__version__,
    transformers=transformers.__version__, threads=torch.get_num_threads())

t0 = time.time()
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32,
                                               safety_checker=None)
pipe.set_progress_bar_config(disable=True)
log(stage="load", seconds=round(time.time() - t0, 1))

CASES = [(256, 1), (384, 1), (512, 1), (512, 2), (512, 4)]
for size, steps in CASES:
    for rep in range(2):
        g = torch.Generator("cpu").manual_seed(101)
        t = time.time()
        img = pipe(PROMPT, num_inference_steps=steps, guidance_scale=0.0, height=size, width=size,
                   generator=g).images[0]
        dt = time.time() - t
        log(stage="gen", size=size, steps=steps, rep=rep, seconds=round(dt, 1))
        if rep == 1:
            img.save(f"sdturbo_{size}_{steps}.png")
log(stage="done")
