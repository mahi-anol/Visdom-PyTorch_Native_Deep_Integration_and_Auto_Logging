from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
import visdom
import torch

class SimpleVisdomLogger(Logger):
    def __init__(self, env_name="main", port=8097):
        super().__init__()
        self.vis = visdom.Visdom(port=port, env=env_name)
        self.windows = {}  # Keep track of initialized windows

    @property
    def name(self):
        return "Custom_Lightning_Visdom_Logger"

    @property
    def version(self):
        return "1.0"

    # Expose the underlying Visdom object
    @property
    def experiment(self):
        return self.vis

    @rank_zero_only
    def log_hyperparams(self, params):
        params_str = str(params)
        self.vis.text(params_str, win="hyperparams", opts=dict(title="Hyperparameters"))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # Fallback if step is somehow None (Lightning sometimes does this at epoch end)
        if step is None:
            step = 0 

        for name, value in metrics.items():
            # Skip non-numeric metrics (like strings) that Visdom's line plot can't handle
            if isinstance(value, str):
                continue

            if torch.is_tensor(value):
                value = value.item()
            
            self.vis.line(
                X=[step],
                Y=[value],
                win=name,
                update='append' if name in self.windows else None,
                opts=dict(title=name, xlabel='Step', ylabel='Value')
            )
            self.windows[name] = True