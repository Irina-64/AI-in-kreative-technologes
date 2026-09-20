"""Демонстрация к лекции 6: диффузионная модель на плоскости (три перекрывающихся кластера).
Запуск: python lec06_demo.py   Вывод: demo6.json (метрики), fig_toy.png (панели с выборками), demo6_log.txt
Что измеряется: прямой процесс (расписание), предсказание x0 по шагам, качество сэмплера от числа шагов
(DDIM) и против вероятностного сэмплера DDPM, безклассификаторное управление (s = 1...8).
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(min(4, torch.get_num_threads()))

T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)          # линейное расписание из работы Ho и др. (2020)
ALPHAS = 1.0 - BETAS
ABAR = torch.cumprod(ALPHAS, 0)
R, SIG = 1.5, 0.55
ANG = torch.tensor([90.0, 210.0, 330.0]) * math.pi / 180
CENTERS = torch.stack([R * torch.cos(ANG), R * torch.sin(ANG)], 1)   # 3 x 2
NULL = 3                                       # индекс «пустого» условия
ITERS, BATCH, P_DROP = int(os.environ.get("ITERS", 8000)), 512, 0.1
N_EVAL = 3000                                  # по 1000 точек на класс
MODEL_SEEDS = (101, 102, 103)
SAMPLE_SEEDS = (101, 102, 103)
LOG = open("demo6_log.txt", "w", encoding="utf-8")


def log(*a):
    print(*a, file=LOG, flush=True)


def make_data(n, gen):
    c = torch.randint(0, 3, (n,), generator=gen)
    x = CENTERS[c] + SIG * torch.randn(n, 2, generator=gen)
    return x, c


def t_features(t):
    tt = t.float().unsqueeze(1) / T
    k = torch.arange(1, 9).float().unsqueeze(0)
    return torch.cat([tt, torch.sin(2 * math.pi * k * tt), torch.cos(2 * math.pi * k * tt)], 1)


class Net(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 + 17 + 4, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t, c):
        oh = torch.nn.functional.one_hot(c, 4).float()
        return self.f(torch.cat([x, t_features(t), oh], 1))


def train(seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        x0, c = make_data(BATCH, gen)
        c = torch.where(torch.rand(BATCH, generator=gen) < P_DROP, torch.full_like(c, NULL), c)
        t = torch.randint(0, T, (BATCH,), generator=gen)
        eps = torch.randn(BATCH, 2, generator=gen)
        ab = ABAR[t].unsqueeze(1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = ((net(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0 or it == ITERS - 1:
            log("seed", seed, "iter", it, "loss", round(loss.item(), 4))
    return net.eval()


@torch.no_grad()
def eps_guided(net, x, t, c, s):
    tt = torch.full((x.shape[0],), t, dtype=torch.long)
    e_c = net(x, tt, c)
    if s == 1.0:
        return e_c
    e_u = net(x, tt, torch.full_like(c, NULL))
    return e_u + s * (e_c - e_u)


@torch.no_grad()
def ddim(net, xT, c, steps, s):
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    x = xT
    for i, t in enumerate(ts):
        eps = eps_guided(net, x, int(t), c, s)
        ab = ABAR[t]
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        ab_prev = ABAR[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x


@torch.no_grad()
def ddpm(net, xT, c, gen):
    x = xT
    for t in range(T - 1, -1, -1):
        eps = eps_guided(net, x, t, c, 1.0)
        a, ab, b = ALPHAS[t], ABAR[t], BETAS[t]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        x = mean + (b.sqrt() * torch.randn(x.shape, generator=gen) if t > 0 else 0.0)
    return x


def energy(x, y):
    dxy = torch.cdist(x, y).mean()
    dxx = torch.cdist(x, x).sum() / (len(x) * (len(x) - 1))
    dyy = torch.cdist(y, y).sum() / (len(y) * (len(y) - 1))
    return (2 * dxy - dxx - dyy).item()


def purity(x, c):
    return (torch.cdist(x, CENTERS).argmin(1) == c).float().mean().item()


def spread(x, c):
    return (x - CENTERS[c]).norm(dim=1).mean().item()


def mstd(v):
    a = np.array(v, dtype=float)
    return [round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)]


def sample_ddim(net, seed, steps, s):
    gen = torch.Generator().manual_seed(seed)
    xT = torch.randn(N_EVAL, 2, generator=gen)
    c = torch.arange(3).repeat_interleave(N_EVAL // 3)
    t0 = time.time()
    x = ddim(net, xT, c, steps, s)
    return x, c, time.time() - t0


def main():
    out = {"meta": {"torch": torch.__version__, "threads": torch.get_num_threads(), "T": T,
                    "beta_range": [1e-4, 0.02], "iters": ITERS, "batch": BATCH, "p_drop": P_DROP,
                    "R": R, "sigma": SIG, "n_eval": N_EVAL,
                    "model_seeds": MODEL_SEEDS, "sample_seeds": SAMPLE_SEEDS}}
    # 1. расписание (арифметика): ab_t, доля сигнала, ОСШ в дБ
    rows = []
    for t in (1, 50, 100, 250, 500, 750, 1000):
        ab = ABAR[t - 1].item()
        rows.append({"t": t, "abar": round(ab, 5), "sqrt_abar": round(math.sqrt(ab), 4),
                     "snr_db": round(10 * math.log10(ab / (1 - ab)), 2)})
    out["schedule"] = rows
    # 2. эталоны: шум оценки энергии между двумя независимыми выборками истинных данных и чистота истинных данных
    g = torch.Generator().manual_seed(7)
    cls =torch.arange(3).repeat_interleave(N_EVAL // 3)
    real = torch.cat([CENTERS[i] + SIG * torch.randn(N_EVAL // 3, 2, generator=g) for i in range(3)])
    real2 = torch.cat([CENTERS[i] + SIG * torch.randn(N_EVAL // 3, 2, generator=g) for i in range(3)])
    ref_real = real
    out["reference"] = {"energy_floor": round(energy(real, real2), 4), "purity_true": round(purity(real, cls), 4),
                        "spread_true": round(spread(real, cls), 4)}
    res_steps, res_cfg, res_x0, res_ddpm, times = {}, {}, {}, [], {}
    nets = {}
    for ms in MODEL_SEEDS:
        t0 = time.time(); nets[ms] = train(ms); log("train", ms, round(time.time() - t0, 1), "s")
        out.setdefault("train_seconds", []).append(round(time.time() - t0, 1))
    STEPS = (1, 2, 3, 5, 10, 25, 50, 100)
    SCALES = (1.0, 2.0, 3.0, 5.0, 8.0)
    for ms, net in nets.items():
        # 3. качество от числа шагов (DDIM, без управления)
        for st in STEPS:
            e = []
            for ss in SAMPLE_SEEDS:
                x, c, dt = sample_ddim(net, ss, st, 1.0)
                e.append(energy(x, ref_real))
                times.setdefault(f"ddim{st}", []).append(dt)
            res_steps.setdefault(st, []).append(float(np.mean(e)))
        # 4. управление
        for s in SCALES:
            p, sp, e = [], [], []
            for ss in SAMPLE_SEEDS:
                x, c, dt = sample_ddim(net, ss, 50, s)
                p.append(purity(x, c)); sp.append(spread(x, c)); e.append(energy(x, ref_real))
            res_cfg.setdefault(s, {"purity": [], "spread": [], "energy": []})
            res_cfg[s]["purity"].append(float(np.mean(p)))
            res_cfg[s]["spread"].append(float(np.mean(sp)))
            res_cfg[s]["energy"].append(float(np.mean(e)))
        # 5. вероятностный сэмплер DDPM, 1000 шагов
        gen = torch.Generator().manual_seed(101)
        xT = torch.randn(N_EVAL, 2, generator=gen)
        c = torch.arange(3).repeat_interleave(N_EVAL // 3)
        t0 = time.time(); x = ddpm(net, xT, c, gen); dt = time.time() - t0
        res_ddpm.append((energy(x, ref_real), purity(x, c), dt))
        # 6. воспроизводимость DDIM: два запуска с одним шумом
        xa, _, _ = sample_ddim(net, 101, 50, 1.0); xb, _, _ = sample_ddim(net, 101, 50, 1.0)
        out.setdefault("ddim_repeat_maxdiff", []).append((xa - xb).abs().max().item())
        # 7. ошибка оценки x0 по шагам на истинных данных (условная модель, s = 1)
        gt = torch.Generator().manual_seed(11)
        x0, c0 = make_data(4000, gt)
        for t in (10, 100, 250, 500, 750, 990):
            eps = torch.randn(4000, 2, generator=gt)
            ab = ABAR[t]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            with torch.no_grad():
                ep = net(xt, torch.full((4000,), t, dtype=torch.long), c0)
            xh = (xt - (1 - ab).sqrt() * ep) / ab.sqrt()
            r_model = (xh - x0).norm(dim=1).pow(2).mean().sqrt().item()
            r_center = (CENTERS[c0] - x0).norm(dim=1).pow(2).mean().sqrt().item()
            r_noisy = (xt / ab.sqrt() - x0).norm(dim=1).pow(2).mean().sqrt().item()
            res_x0.setdefault(t, {"model": [], "no_denoise": []})
            res_x0[t]["model"].append(r_model); res_x0[t]["no_denoise"].append(r_noisy)
            res_x0[t]["center"] = round(r_center, 4)
    out["steps_energy"] = {str(k): mstd(v) for k, v in res_steps.items()}
    out["steps_seconds"] = {k: round(float(np.mean(v)), 3) for k, v in times.items()}
    out["cfg"] = {str(k): {m: mstd(v[m]) for m in v} for k, v in res_cfg.items()}
    out["ddpm1000"] = {"energy": mstd([r[0] for r in res_ddpm]), "purity": mstd([r[1] for r in res_ddpm]),
                       "seconds": round(float(np.mean([r[2] for r in res_ddpm])), 2)}
    out["x0_rmse"] = {str(k): {"model": mstd(v["model"]), "no_denoise": mstd(v["no_denoise"]), "center": v["center"]}
                      for k, v in res_x0.items()}
    with open("demo6.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    # рисунок: истинные данные и выборки для s = 1, 3, 8 (модель 101, шум 101)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.4), dpi=80)
    cols = ["#c0392b", "#27ae60", "#2980b9"]
    panels = [("данные", real, cls)]
    for s in (1.0, 3.0, 8.0):
        x, c, _ = sample_ddim(nets[101], 101, 50, s)
        panels.append((f"s = {s:g}", x, c))
    for ax, (title, x, c) in zip(axs, panels):
        for k in range(3):
            m = c == k
            ax.scatter(x[m, 0], x[m, 1], s=1.5, color=cols[k])
        ax.scatter(CENTERS[:, 0], CENTERS[:, 1], marker="x", color="k", s=25)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig("fig_toy.png")
    print("OK", flush=True)


if __name__ == "__main__":
    main()
