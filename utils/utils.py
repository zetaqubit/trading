"""Common utility functions."""

def map_structure(f, struct):
  if isinstance(struct, list):
    return [map_structure(f, item) for item in struct]
  if isinstance(struct, tuple):
    return tuple(map_structure(f, item) for item in struct)
  if isinstance(struct, dict):
    return {k: map_structure(f, v) for k, v in struct.items()}
  return f(struct)


def split_xy(struct, y_len):
  """Splits Tensors in the structure into x and y.

  Struct is a possibly nested list/tuple/dict of Tensors. This
  function splits each Tensor into [:-y_len] and [-y_len:]
  parts, returning 2 separate structures.
  """
  def x(ary):
    return ary[:-y_len]
  def y(ary):
    return ary[-y_len:]
  return map_structure(x, struct), map_structure(y, struct)

