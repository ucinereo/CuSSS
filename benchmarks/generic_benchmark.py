import torch


def benchmark_on_cuda(modules : dict[str, torch.nn.Module], tensor_sizes : list[int] = [2**10, 2**14, 2**18, 2**22]):
    """
    Generic benchmark function which takes some modules (here for the activation functions) and records the time 
    it takes to apply the forward and backward functions each 100 times. On cuda-device.
    """
    device = torch.device("cuda")

    # Iterate over tensor sizes
    for size in tensor_sizes:

        batch_size = 64
        x = torch.randn(batch_size, size, device=device, requires_grad=True)

        print(f"[Tensor size ({batch_size}, {size})]:")

        for module_name in modules:
            activ_fn = modules[module_name].to(device)

            # Forward pass:
            # Warm-up
            for _ in range(10):
                y = activ_fn(x)

            # Measure forward passes
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                y = activ_fn(x)
            end.record()

            torch.cuda.synchronize()
            print(f"[{module_name}]\t Forward time (ms):", start.elapsed_time(end))


            # Backward pass:
            loss = y.sum()

            # Warm-up
            for _ in range(10):
                loss.backward(retain_graph=True)

            # Measure backward passes
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                loss.backward(retain_graph=True)
            end.record()

            torch.cuda.synchronize()
            print(f"[{module_name}]\t Backward time (ms):", start.elapsed_time(end))

    return

