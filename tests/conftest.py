def odd_shape(shape): # defined as when all dimension is not power of 2 and greater than 1
    return all(dim % 4 != 0 and dim > 1 for dim in shape)

def pytest_addoption(parser):
    parser.addoption(
        "--kernel",
        action="store",
        default=None,
        help="Run tests only for specified kernel (e.g., sss, xsss)",
    )
    parser.addoption(
        "--shape",
        action="store",
        default="even", # this repo only handles divisible by 4 shapes
        help="Run tests only for even or odd inputs (odd shape has all dimensions divisible by 4 and greater than 1)",
    )
    parser.addoption(
        "--loss",
        action="store",
        default=None,
        help="Run tests only for specified loss function {sum, mean, l2, mse}",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "kernel(name): mark test for a specific kernel")


def pytest_collection_modifyitems(config, items):
    shapes_filter = config.getoption("--shape")
    if shapes_filter:
        remaining = []
        for item in items:
            if hasattr(item, "callspec") and "shape" in item.callspec.params:
                shape = item.callspec.params["shape"]
                if shapes_filter == "odd" and not odd_shape(shape):
                    continue
                if shapes_filter == "even" and odd_shape(shape):
                    continue
            remaining.append(item)
        items[:] = remaining

    
    kernel_filter = config.getoption("--kernel")
    if kernel_filter:
        remaining = []
        for item in items:
            # Check parametrized kernel name
            if hasattr(item, "callspec") and "kernel_name" in item.callspec.params:
                if item.callspec.params["kernel_name"] == kernel_filter:
                    remaining.append(item)
            # Check marker
            elif item.get_closest_marker("kernel"):
                marker = item.get_closest_marker("kernel")
                if marker.args[0] == kernel_filter:
                    remaining.append(item)
        items[:] = remaining

    loss_filter = config.getoption("--loss")
    if loss_filter:
        remaining = []
        for item in items:
            if hasattr(item, "callspec") and "loss_name" in item.callspec.params:
                if item.callspec.params["loss_name"] == loss_filter:
                    remaining.append(item)
        items[:] = remaining