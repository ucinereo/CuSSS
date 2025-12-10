def pytest_addoption(parser):
    parser.addoption(
        "--kernel",
        action="store",
        default=None,
        help="Run tests only for specified kernel (e.g., sss, xsss)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "kernel(name): mark test for a specific kernel")


def pytest_collection_modifyitems(config, items):
    kernel_filter = config.getoption("--kernel")
    if kernel_filter is None:
        return

    selected = []
    for item in items:
        # Check parametrized kernel name
        if hasattr(item, "callspec") and "kernel_name" in item.callspec.params:
            if item.callspec.params["kernel_name"] == kernel_filter:
                selected.append(item)
        # Check marker
        elif item.get_closest_marker("kernel"):
            marker = item.get_closest_marker("kernel")
            if marker.args[0] == kernel_filter:
                selected.append(item)

    items[:] = selected
