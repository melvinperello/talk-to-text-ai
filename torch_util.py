"""torch_util.py"""

import subprocess

import torch


def get_torch_device():
    """Detect and return the available PyTorch device type.

    Returns:
        str: Device type - 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon,
            or raises ValueError if neither is available.

    Raises:
        ValueError: If neither CUDA nor MPS is available on the machine.
    """
    if torch.cuda.is_available():
        print("CUDA is available!  Let's use it.")
        return "cuda"
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("MPS is available!  Let's use it.")
            return "mps"
        else:
            raise ValueError(
                "MPS is not available because the current PyTorch install was not built with MPS enabled."
            )
    else:
        raise ValueError("Neither CUDA nor MPS is available on this machine.")


def get_torch_device_name():
    """Get the human-readable name of the current PyTorch device.

    Returns:
        str: Device name - 'Apple M1', 'Apple M2', etc. for MPS,
            or GPU name for CUDA devices.

    Raises:
        ValueError: If device type is unsupported.
    """
    t_device = get_torch_device()
    if t_device == "cuda":
        return torch.cuda.get_device_name()
    if t_device == "mps":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    raise ValueError("Unsupported device type.")


if __name__ == "__main__":
    device_name = get_torch_device_name()
    print(f"Torch Util Device: {device_name}")
