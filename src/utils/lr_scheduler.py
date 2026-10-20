from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def get_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=1000, start_factor=0.01, min_lr=1e-6):
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
    )
    if warmup_steps == 0:
        return cosine

    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    return scheduler
