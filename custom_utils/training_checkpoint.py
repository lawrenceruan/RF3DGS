"""Atomic checkpoints for all layers and random streams of the layered trainer."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_training_checkpoint(path, models, iteration, cameras, remaining,
                             optimizer_step_pending=False):
    path = Path(path)
    state = {
        "format_version": 1,
        "iteration": iteration,
        "models": [model.capture() for model in models],
        "camera_order": [cam.image_name for cam in cameras],
        "remaining": [cam.image_name for cam in remaining],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "optimizer_step_pending": optimizer_step_pending,
        "gradients": [[p.grad for group in model.optimizer.param_groups
                       for p in group["params"]] for model in models]
                     if optimizer_step_pending else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_training_checkpoint(path, models, options, cameras):
    # This entry point reads a local training checkpoint, including RNG tuples.
    state = torch.load(path, weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Legacy checkpoint contains only one overwritten layer; resume requires a complete layered checkpoint")
    by_name = {cam.image_name: cam for cam in cameras}
    if len(by_name) != len(cameras) or set(by_name) != set(state["camera_order"]):
        raise ValueError("Checkpoint camera set differs from the training data")
    if len(state["models"]) != len(models):
        raise ValueError("Checkpoint must contain all three Gaussian layers")
    for model, saved in zip(models, state["models"]):
        model.restore(saved, options)
    if state["optimizer_step_pending"]:
        for model, gradients in zip(models, state["gradients"]):
            parameters = [p for group in model.optimizer.param_groups for p in group["params"]]
            if len(parameters) != len(gradients):
                raise ValueError("Checkpoint gradient count differs from the optimizer")
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
    cameras[:] = [by_name[name] for name in state["camera_order"]]
    remaining = [by_name[name] for name in state["remaining"]]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["iteration"], remaining
