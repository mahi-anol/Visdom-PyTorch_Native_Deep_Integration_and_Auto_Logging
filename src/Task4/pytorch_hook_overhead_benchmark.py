import torch
import torch.nn as nn
import time
from torch.profiler import profile, record_function, ProfilerActivity
from src.Task1.cnn_model import SimpleCNN
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.profiler")

def profile_cnn_hooks():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    dummy_input = torch.randn(64, 1, 28, 28).to(device)
    dummy_target = torch.randint(0, 10, (64,)).to(device)

    criterion = nn.CrossEntropyLoss()

    # Latency Profiling in manual way.
    def run_benchmark(iterations=50):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / iterations

    avg_no_hooks = run_benchmark()
    
    # Register Hooks
    hooks = []
    def gradient_logging_hook(grad):
        # We wrap this in a record_function so the profiler catches it specifically
        with record_function("visdom_hook_execution"):
            _ = grad.detach().norm(2) 
        return grad

    for param in model.parameters():
        if param.requires_grad:
            hooks.append(param.register_hook(gradient_logging_hook))

    avg_with_hooks = run_benchmark()
    overhead = (avg_with_hooks / avg_no_hooks - 1) * 100
    print(f"Total Overhead: {overhead:.2f}%")

    # Using Torch Profiler 
    print("\nStarting Torch Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("batch_pass_with_hooks"):
            model.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()


    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    for h in hooks: h.remove()

if __name__ == "__main__":
    profile_cnn_hooks()