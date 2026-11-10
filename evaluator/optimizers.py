from __future__ import annotations

from typing import Any, Dict

import torch


def build_optimizer(model, optimizer_name: str = "adamw", learning_rate: float = 2e-4, betas=(0.8, 0.99), eps: float = 1e-8):
    """Create an optimizer for a network."""
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=eps)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, lr_decay: float = 0.999875):
    """Match the original VITS learning-rate decay strategy."""
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)


def build_optimizers(generator, discriminator, config: Dict[str, Any]):
    """Create optimizers for generator and discriminator."""
    train_cfg = config.get("training", {})
    optimizer_cfg = train_cfg.get("optimizer", {})
    lr = train_cfg.get("learning_rate", 2e-4)
    betas = tuple(train_cfg.get("betas", (0.8, 0.99)))
    eps = train_cfg.get("eps", 1e-8)
    lr_decay = train_cfg.get("lr_decay", 0.999875)

    optimizer_name = optimizer_cfg.get("name", "adamw")
    opt_g = build_optimizer(generator, optimizer_name, lr, betas, eps)
    opt_d = build_optimizer(discriminator, optimizer_name, lr, betas, eps)

    return {
        "generator": opt_g,
        "discriminator": opt_d,
        "generator_scheduler": build_scheduler(opt_g, lr_decay),
        "discriminator_scheduler": build_scheduler(opt_d, lr_decay),
    }
