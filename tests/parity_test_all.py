from xml.parsers.expat import model
import torch
import pytest
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, Any

# custom kernels
from cusss import SSS
from cusss import xSSS
# **import new modules here if needed**


# mirrored pytorch kernels
from pytorch_kernels import sss, sss_ref_forward, sss_ref_backward
from pytorch_kernels import xsss, xsss_ref_forward, xsss_ref_backward
# **go into tests/pytorch_kernels.py to add new mirrored pytorch kernels**


@dataclass
class KernelConfig:
    name: str
    custom_kernel: Callable        # The custom CUDA module class (e.g., SSS)
    pytorch_kernel: Callable       # The PyTorch mirrored module class (e.g., sss)
    gt_fn: Callable                # The ground truth function for forward pass
    gt_grad_fn: Callable           # The ground truth gradient function for backward pass
    input_shape: Tuple[int, ...] = (64, 512)
    inputs: dict = field(default_factory=lambda: {"x"}) # Which inputs to test (x, y, a)
    dtype: torch.dtype = torch.float32
    test_params: dict = field(default_factory=dict) # Args for module __init__



# --- THE REGISTRY ---
# Add new test cases here. Pytest will automatically spawn a test for each.
TEST_CASES = [
    KernelConfig(
        name="SSS_Standard_fp32",
        custom_kernel=SSS,
        pytorch_kernel=sss,
        gt_fn=sss_ref_forward,
        gt_grad_fn=sss_ref_backward,
        input_shape=(64, 512),
    ),
    KernelConfig(
        name="SSS_Small_Batch",
        custom_kernel=SSS,
        pytorch_kernel=sss,
        gt_fn=sss_ref_forward,
        gt_grad_fn=sss_ref_backward,
        input_shape=(1, 128) # Test edge case: batch size 1
    ),
    # KernelConfig(
    #     name="SSS_Odd_Shape",
    #     module_cls=SSS,
    #     gt_fn=sss_ref_forward,
    #     ref_grad_fn=sss_ref_backward,
    #     input_shape=(33, 513) # Test edge case: non-power-of-2 shapes
    # ),
    KernelConfig(
        name="xSSS_Standard_fp32",
        custom_kernel=xSSS,
        pytorch_kernel=xsss,
        gt_fn=xsss_ref_forward,
        gt_grad_fn=xsss_ref_backward,
        input_shape=(64, 512),
        inputs={"x", "a"},
    ),
    KernelConfig(
        name="xSSS_Small_Batch",
        custom_kernel=xSSS,
        pytorch_kernel=xsss,
        gt_fn=xsss_ref_forward,
        gt_grad_fn=xsss_ref_backward,
        input_shape=(1, 128), # Test edge case: batch size 1
        inputs={"x", "a"},
    ),
    # **add new test cases here as needed**
]

@pytest.mark.parametrize("config", TEST_CASES, ids=lambda c: c.name)
class TestCudaKernels:

    @pytest.fixture
    def setup_vars(self, config):
        """Fixture to handle device setup and cleanup."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        torch.manual_seed(42) # Ensure determinism
        device = torch.device("cuda")
        
        # Init module
        custom_model = config.custom_kernel(**config.test_params).to(device, dtype=config.dtype)
        pytorch_model = config.pytorch_kernel
        
        # Init input
        x = torch.randn(
            config.input_shape, 
            device=device, 
            dtype=config.dtype, 
            requires_grad=True
        )
        y = torch.randn(
            config.input_shape, 
            device=device, 
            dtype=config.dtype, 
            requires_grad=True
        )
        a = torch.randn(1, 
            device=device, 
            dtype=config.dtype, 
            requires_grad=True
        )
        
        return {
            "x": x,
            "y": y,
            "a": a,
            "custom_model": custom_model,
            "pytorch_model": pytorch_model
        }

    def test_forward(self, config, setup_vars):
        vars = setup_vars
        input = {k: vars[k] for k in config.inputs}
        custom_model = vars["custom_model"]
        pytorch_model = vars["pytorch_model"]
        
        # Run Custom Kernel
        y_custom = custom_model(**input)

        # Run PyTorch Kernel        
        y_pytorch = pytorch_model(**input)
        
        
        # Run Reference
        y_ref = config.gt_fn(**input)
        
        # 3. Compare
        torch.testing.assert_close(y_custom, y_ref, msg=f"Forward mismatch (ref): {config.name}")
        torch.testing.assert_close(y_custom, y_pytorch, msg=f"Forward mismatch (pytorch): {config.name}")

    def test_backward(self, config, setup_vars):
            vars = setup_vars
            input_keys = config.inputs
            
            # helper
            def compute_gradients(forward_fn, model_name):
                local_inputs = {
                    k: vars[k].clone().detach().requires_grad_(True) 
                    for k in input_keys
                }
                
                
                # forward
                try:
                    y = forward_fn(**local_inputs)
                except Exception as e:
                    pytest.fail(f"Forward pass failed for {model_name}: {e}")

                # Backward Pass
                # Simple toy loss 
                loss = y.sum()
                loss.backward()
                
                grads = {k: local_inputs[k].grad for k in input_keys}
                return grads

            # Run Custom Kernel 
            grads_custom = compute_gradients(vars["custom_model"], "Custom Kernel")

            # Run PyTorch Mirrored Kernel
            grads_pytorch = compute_gradients(vars["pytorch_model"], "PyTorch Kernel")

            # Run Reference (Ground Truth)
            # We use autograd on the gt_fn to ensure we get general gradients 
            # for x, y, a, etc., without relying on hardcoded gt_grad_fn signatures.
            grads_ref = config.gt_grad_fn(**{k: vars[k].detach() for k in input_keys})
            # handle single vs multiple outputs
            if len(input_keys) == 1:
                grads_ref = {list(input_keys)[0]: grads_ref}
            else:
                grads_ref = {
                    k: grads_ref[i] 
                    for i, k in enumerate(input_keys)
                }
            
            # --- 4. Compare ---
            for k in input_keys:
                # Retrieve gradients for variable k
                g_custom = grads_custom[k]
                g_pytorch = grads_pytorch[k]
                g_ref = grads_ref[k]

                # Ensure gradient computation worked 
                assert g_custom is not None, f"Custom kernel returned None grad for input '{k}'"
                assert g_pytorch is not None, f"PyTorch kernel returned None grad for input '{k}'"
                assert g_ref is not None, f"Reference returned None grad for input '{k}'"

                # Compare Custom vs Reference
                # torch.testing.assert_close(
                #     g_custom, 
                #     g_ref, 
                #     msg=f"Backward mismatch (ref) for var '{k}': {config.name}"
                # )
                
                # Compare Custom vs PyTorch Mirror
                torch.testing.assert_close(
                    g_custom, 
                    g_pytorch, 
                    msg=f"Backward mismatch (pytorch) for var '{k}': {config.name}"
                )