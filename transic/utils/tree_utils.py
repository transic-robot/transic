from typing import List, TypeVar, Iterable, Tuple
import numpy as np

try:
    import tree

except ImportError:
    raise ImportError("Please install dm_tree first: `pip install dm_tree`")

ElementType = TypeVar("ElementType")


def fast_map_structure(func, *structure):
    """Faster map_structure implementation which skips some error checking."""
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def stack_sequence_fields(sequence: Iterable[ElementType]) -> ElementType:
    """Stacks a list of identically nested objects.

    This takes a sequence of identically nested objects and returns a single
    nested object whose ith leaf is a stacked numpy array of the corresponding
    ith leaf from each element of the sequence.

    For example, if `sequence` is:

    ```python
    [{
          'action': np.array([1.0]),
          'observation': (np.array([0.0, 1.0, 2.0]),),
          'reward': 1.0
     }, {
          'action': np.array([0.5]),
          'observation': (np.array([1.0, 2.0, 3.0]),),
          'reward': 0.0
     }, {
          'action': np.array([0.3]),1
          'observation': (np.array([2.0, 3.0, 4.0]),),
          'reward': 0.5
     }]
    ```

    Then this function will return:

    ```python
    {
        'action': np.array([....])         # array shape = [3 x 1]
        'observation': (np.array([...]),)  # array shape = [3 x 3]
        'reward': np.array([...])          # array shape = [3]
    }
    ```

    Note that the 'observation' entry in the above example has two levels of
    nesting, i.e it is a tuple of arrays.

    Args:
      sequence: a list of identically nested objects.

    Returns:
      A nested object with numpy.

    Raises:
      ValueError: If `sequence` is an empty sequence.
    """
    # Handle empty input sequences.
    if not sequence:
        raise ValueError("Input sequence must not be empty")

    # Default to asarray when arrays don't have the same shape to be compatible
    # with old behaviour.
    try:
        return fast_map_structure(lambda *values: np.stack(values), *sequence)
    except ValueError:
        return fast_map_structure(lambda *values: np.asarray(values), *sequence)


def unstack_sequence_fields(struct: ElementType, batch_size: int) -> List[ElementType]:
    """Converts a struct of batched arrays to a list of structs.

    This is effectively the inverse of `stack_sequence_fields`.

    Args:
      struct: An (arbitrarily nested) structure of arrays.
      batch_size: The length of the leading dimension of each array in the struct.
        This is assumed to be static and known.

    Returns:
      A list of structs with the same structure as `struct`, where each leaf node
       is an unbatched element of the original leaf node.
    """

    return [tree.map_structure(lambda s, i=i: s[i], struct) for i in range(batch_size)]


def tree_value_at_path(obj, paths: Tuple):
    try:
        for p in paths:
            obj = obj[p]
        return obj
    except Exception as e:
        raise ValueError(f"{e}\n\n-- Incorrect nested path {paths} for object: {obj}.")
