""Input tensors in BCHW. Real datasets if torchvision + internet exist,
otherwise clearly-labelled synthetic stand-ins of identical shape.""
import numpy as np


def _try_torchvision(name):
    try:
        import torchvision
        cls = {"MNIST": torchvision.datasets.MNIST,
               "CIFAR10": torchvision.datasets.CIFAR10}[name]
        ds = cls(root="./data", train=True, download=True)
        img = np.array(ds[0][0], dtype=np.float32) / 255.0
        if img.ndim == 2:                       # MNIST: H x W
            img = img[None, :, :]               # -> C x H x W
        else:                                   # CIFAR: H x W x C
            img = np.transpose(img, (2, 0, 1))  # -> C x H x W
        return img[None, ...]                   # -> 1 x C x H x W
    except Exception:
        return None


def load_mnist_sample(seed=0):
    t = _try_torchvision("MNIST")
    if t is not None:
        return t, "MNIST / sample"
    rng = np.random.default_rng(seed)
    return rng.random((1, 1, 28, 28), dtype=np.float32), "MNIST-shaped (synthetic)"


def load_cifar_sample(seed=1):
    t = _try_torchvision("CIFAR10")
    if t is not None:
        return t, "CIFAR-10 / sample"
    rng = np.random.default_rng(seed)
    return rng.random((1, 3, 32, 32), dtype=np.float32), "CIFAR-shaped (synthetic)"


def synthetic_fmap(B, C, H, W, seed=42):
    """Random feature map, like the output of an intermediate DNN layer."""
    rng = np.random.default_rng(seed + C)
    return rng.standard_normal((B, C, H, W)).astype(np.float32)
