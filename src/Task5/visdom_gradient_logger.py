import torch
import visdom

class VisdomGradientLogger:
    def __init__(self, model, env_name="gradient_monitor_visdom", port=8097):
        self.vis = visdom.Visdom(port=port, env=env_name)
        self.model = model
        self.hooks = []
        self.step = 0
        self.windows = {}
        self._grad_buffer = {} # Buffer to prevent network spam
        self._attach_hooks()

    def _attach_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Capture the name and attach
                hook = param.register_hook(lambda grad, n=name: self._buffer_grad_norm(grad, n))
                self.hooks.append(hook)
        print(f"Attached hooks to {len(self.hooks)} parameters.")

    def _buffer_grad_norm(self, grad, name):
        """Storing the norm in a buffer instead of sending immediately."""
        self._grad_buffer[name] = grad.norm(2).item()

    def log_step(self):
        """Call this ONCE per batch. It sends all buffered gradients in one go."""
        for name, grad_norm in self._grad_buffer.items():
            win_name = f"grad_{name.replace('.', '_')}"
            
            self.vis.line(
                X=[self.step],
                Y=[grad_norm],
                win=win_name,
                name=name, # Using 'name' inside the window for overlaying plots
                update='append' if win_name in self.windows else None,
                opts=dict(title=f"Grad: {name}", xlabel="Step", ylabel="L2 Norm")
            )
            self.windows[win_name] = True
        
        self.step += 1
        self._grad_buffer.clear() # Clear for next batch

    def cleanup(self):
        for h in self.hooks: h.remove()