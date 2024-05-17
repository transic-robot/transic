import math
import torch
import numpy as np


def generate_cosine_schedule(
    base_value,
    final_value,
    epochs,
    steps_per_epoch,
    warmup_epochs=0,
    warmup_start_value=0,
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * steps_per_epoch)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_start_value, base_value, warmup_iters)

    iters = np.arange(int(epochs * steps_per_epoch) - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == int(epochs * steps_per_epoch)
    return schedule


class CosineScheduleFunction:
    def __init__(
        self,
        base_value,
        final_value,
        epochs,
        steps_per_epoch,
        warmup_epochs=0,
        warmup_start_value=0,
    ):
        """
        Usage:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=CosineScheduleFunction(**kwargs)
            )
            or simply use CosineScheduler(**kwargs)

        Args:
            epochs: effective epochs for the cosine schedule, *including* warmup
                after these epochs, scheduler will output `final_value` ever after
        """
        assert warmup_epochs < epochs, f"{warmup_epochs=} must be < {epochs=}"
        self._effective_steps = int(epochs * steps_per_epoch)
        self.schedule = generate_cosine_schedule(
            base_value=base_value,
            final_value=final_value,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            warmup_start_value=warmup_start_value,
        )
        assert self.schedule.shape == (self._effective_steps,)
        self._final_value = final_value
        self._steps_tensor = torch.tensor(0, dtype=torch.long)  # for register buffer

    def register_buffer(self, module: torch.nn.Module, name="cosine_steps"):
        module.register_buffer(name, self._steps_tensor, persistent=True)

    def __call__(self, step):
        self._steps_tensor.copy_(torch.tensor(step))
        if step >= self._effective_steps:
            val = self._final_value
        else:
            val = self.schedule[step]
        return val
