try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True

try:
    import torch
    from torch.autograd import Variable

    torch_available = True
except ImportError as e:
    torch_available = False

try:
    import jax
    import jax.numpy as jnp

    jax_available = True
except ImportError as e:
    jax_available = False


# pylint: disable=unused-argument
def pytest_generate_tests(metafunc):
    if jax_available:
        jax.config.update("jax_enable_x64", True)


def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    # Autograd is assumed to be installed
    interfaces = {"tf", "torch", "jax"}
    available_interfaces = {
        "tf": tf_available,
        "torch": torch_available,
        "jax": jax_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf", "torch", "jax", "all_interfaces"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b == "all_interfaces":
            required_interfaces = {"tf", "torch", "jax"}
            for interface in required_interfaces:
                if interface not in allowed_interfaces:
                    pytest.skip(
                        f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                    )
        else:
            if b not in allowed_interfaces:
                pytest.skip(
                    f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                )