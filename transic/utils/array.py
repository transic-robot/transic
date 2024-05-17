from typing import List, Union, Dict

import functools
import tree
import numpy as np
import torch


__ALL__ = [
    "any_slice",
    "nested_np_split",
    "get_batch_size",
    "any_stack",
    "any_concat",
    "any_to_torch_tensor",
]


_TORCH_DTYPE_TABLE = {
    torch.bool: 1,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.short: 2,
    torch.int32: 4,
    torch.int: 4,
    torch.int64: 8,
    torch.long: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.half: 2,
    torch.float32: 4,
    torch.float: 4,
    torch.float64: 8,
    torch.double: 8,
}


def meta_decorator(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = (
        lambda args, kwargs: len(args) == 1 and len(kwargs) == 0 and callable(args[0])
    )

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(
                lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct
            )
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


@make_recursive_func
def any_slice(x, slice):
    """
    Args:
        slice: you can use np.s_[...] to return the slice object
    """
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x[slice]
    else:
        return x


@make_recursive_func
def nested_np_split(x, indices_or_sections, axis):
    if isinstance(x, np.ndarray):
        return np.split(x, indices_or_sections, axis=axis)
    else:
        raise ValueError(f"Input ({type(x)}) must be a numpy array.")


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif torch.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])


def any_stack(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


def any_concat(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)


@make_recursive_func
def any_ones_like(x: Union[Dict, np.ndarray, torch.Tensor, int, float, np.number]):
    """Returns a one-filled object of the same (d)type and shape as the input.
    The difference between this and `np.ones_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.
    Args:
      x: The object to replace with 1s.
    Returns:
      A one-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(1)
    elif torch.is_tensor(x):
        return torch.ones_like(x)
    elif isinstance(x, np.ndarray):
        return np.ones_like(x)
    else:
        raise ValueError(
            f"Input ({type(x)}) must be either a numpy array, a tensor, an int, or a float."
        )


def _transfer_then_convert(x, dtype, device, copy, non_blocking):
    x = x.to(device=device, copy=copy, non_blocking=non_blocking)
    return x.to(dtype=dtype, copy=False, non_blocking=non_blocking)


def _convert_then_transfer(x, dtype, device, copy, non_blocking):
    x = x.to(dtype=dtype, copy=copy, non_blocking=non_blocking)
    return x.to(device=device, copy=False, non_blocking=non_blocking)


def torch_dtype_size(dtype: Union[str, torch.dtype]) -> int:
    return _TORCH_DTYPE_TABLE[torch_dtype(dtype)]


def torch_device(device: Union[str, int, None]) -> torch.device:
    """
    Args:
        device:
            - "auto": use current torch context device, same as `.to('cuda')`
            - int: negative for CPU, otherwise GPU index
    """
    if device is None:
        return None
    elif device == "auto":
        return torch.device("cuda")
    elif isinstance(device, int) and device < 0:
        return torch.device("cpu")
    else:
        return torch.device(device)


def torch_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f'"{dtype}" is not a valid torch dtype')
        assert isinstance(
            dtype, torch.dtype
        ), f"dtype {dtype} is not a valid torch tensor type"
        return dtype
    else:
        raise NotImplementedError(f"{dtype} not supported")


def any_to_torch_tensor(
    x,
    dtype: Union[str, torch.dtype, None] = None,
    device: Union[str, int, torch.device, None] = None,
    copy=False,
    non_blocking=False,
    smart_optimize: bool = True,
):
    dtype = torch_dtype(dtype)
    device = torch_device(device)

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        # x is a primitive python sequence
        x = torch.tensor(x, dtype=dtype)
        copy = False

    # This step does not create any copy.
    # If x is a numpy array, simply wraps it in Tensor. If it's already a Tensor, do nothing.
    x = torch.as_tensor(x)
    # avoid passing None to .to(), PyTorch 1.4 bug
    dtype = dtype or x.dtype
    device = device or x.device

    if not smart_optimize:
        # do a single stage type conversion and transfer
        return x.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking)

    # we have two choices: (1) convert dtype and then transfer to GPU
    # (2) transfer to GPU and then convert dtype
    # because CPU-to-GPU memory transfer is the bottleneck, we will reduce it as
    # much as possible by sending the smaller dtype

    src_dtype_size = torch_dtype_size(x.dtype)

    # destination dtype size
    if dtype is None:
        dest_dtype_size = src_dtype_size
    else:
        dest_dtype_size = torch_dtype_size(dtype)

    if x.dtype != dtype or x.device != device:
        # a copy will always be performed, no need to force copy again
        copy = False

    if src_dtype_size > dest_dtype_size:
        # better to do conversion on one device (e.g. CPU) and then transfer to another
        return _convert_then_transfer(x, dtype, device, copy, non_blocking)
    elif src_dtype_size == dest_dtype_size:
        # when equal, we prefer to do the conversion on whichever device that's GPU
        if x.device.type == "cuda":
            return _convert_then_transfer(x, dtype, device, copy, non_blocking)
        else:
            return _transfer_then_convert(x, dtype, device, copy, non_blocking)
    else:
        # better to transfer data across device first, and then do conversion
        return _transfer_then_convert(x, dtype, device, copy, non_blocking)


def any_to_numpy(
    x,
    dtype: Union[str, np.dtype, None] = None,
    copy: bool = False,
    non_blocking: bool = False,
    smart_optimize: bool = True,
):
    if isinstance(x, torch.Tensor):
        x = any_to_torch_tensor(
            x,
            dtype=dtype,
            device="cpu",
            copy=copy,
            non_blocking=non_blocking,
            smart_optimize=smart_optimize,
        )
        return x.detach().numpy()
    else:
        # primitive python sequence or ndarray
        return np.array(x, dtype=dtype, copy=copy)
