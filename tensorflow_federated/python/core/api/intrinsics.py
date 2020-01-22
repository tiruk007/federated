# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines intrinsics for use in composing federated computations."""

import warnings

from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_factory


def federated_aggregate(value, zero, accumulate, merge, report):
  """Aggregates `value` from `tff.CLIENTS` to `tff.SERVER`.

  This generalized aggregation function admits multi-layered architectures that
  involve one or more intermediate stages to handle scalable aggregation across
  a very large number of participants.

  The multi-stage aggregation process is defined as follows:

  * Clients are organized into groups. Within each group, a set of all the
    member constituents of `value` contributed by clients in the group are first
    reduced in a manner similar to `tff.federated_reduce` using reduction
    operator `accumulate` with `zero` as the zero in the algebra. As described
    in the documentation for `tff.federated_reduce`, if members of `value` are
    of type `T`, and `zero` (the result of reducing an empty set) is of type
    `U`, the reduction operator `accumulate` used at this stage should be of
    type `(<U,T> -> U)`. The result of this stage is a set of items of type `U`,
    one item for each group of clients.

  * Next, the `U`-typed items generated by the preceding stage are merged using
    the binary commutative associative operator `merge` of type `(<U,U> -> U)`.
    This can be interpreted as a `tff.federated_reduce` using `merge` as the
    reduction operator, and the same `zero` in the algebra. The result of this
    stage is a single top-level `U` that emerges at the root of the hierarchy at
    the `tff.SERVER`. Actual implementations may structure this step as a
    cascade of multiple layers.

  * Finally, the `U`-typed result of the reduction performed in the preceding
    stage is projected into the result value using `report` as the mapping
    function (for example, if the structures being merged consist of counters,
    this final step might include computing their ratios).

  Args:
    value: A value of a TFF federated type placed at `tff.CLIENTS` to aggregate.
    zero: The zero of type `U` in the algebra of reduction operators, as
      described above.
    accumulate: The reduction operator to use in the first stage of the process.
      If `value` is of type `{T}@CLIENTS`, and `zero` is of type `U`, this
      operator should be of type `(<U,T> -> U)`.
    merge: The reduction operator to employ in the second stage of the process.
      Must be of type `(<U,U> -> U)`, where `U` is as defined above.
    report: The projection operator to use at the final stage of the process to
      compute the final result of aggregation. If the intended result to be
      returned by `tff.federated_aggregate` is of type `R@SERVER`, this operator
      must be of type `(U -> R)`.

  Returns:
    A representation on the `tff.SERVER` of the result of aggregating `value`
    using the multi-stage process described above.

  Raises:
    TypeError: if the arguments are not of the types specified above.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_aggregate(value, zero, accumulate, merge, report)


# Deprecated, use tff.federated_map instead.
def federated_apply(fn, arg):
  """Applies a given function to a federated value on `tff.SERVER` (deprecated).

  Args:
    fn: A function to apply to the member content of `arg` on the `tff.SERVER`.
      The parameter of this function must be of the same type as the member
      constituent of `arg`.
    arg: A value of a TFF federated type placed at the `tff.SERVER`, and with
      the `all_equal` bit set.

  Returns:
    A federated value on the `tff.SERVER` that represents the result of applying
    `fn` to the member constituent of `arg`.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  warnings.warn(
      'Deprecation warning: tff.federated_apply() is deprecated, use '
      'tff.federated_map() instead.', DeprecationWarning)
  return federated_map(fn, arg)


def federated_mean(value, weight=None):
  """Computes a `tff.SERVER` mean of `value` placed on `tff.CLIENTS`.

  For values `v_1, ..., v_k`, and weights `w_1, ..., w_k`, this means
  `sum_{i=1}^k (w_i * v_i) / sum_{i=1}^k w_i`.

  Args:
    value: The value of which the mean is to be computed. Must be of a TFF
      federated type placed at `tff.CLIENTS`. The value may be structured, e.g.,
      its member constituents can be named tuples. The tensor types that the
      value is composed of must be floating-point or complex.
    weight: An optional weight, a TFF federated integer or floating-point tensor
      value, also placed at `tff.CLIENTS`.

  Returns:
    A representation at the `tff.SERVER` of the mean of the member constituents
    of `value`, optionally weighted with `weight` if specified (otherwise, the
    member constituents contributed by all clients are equally weighted).

  Raises:
    TypeError: if `value` is not a federated TFF value placed at `tff.CLIENTS`,
      or if `weight` is not a federated integer or a floating-point tensor with
      the matching placement.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_mean(value, weight)


def federated_broadcast(value):
  """Broadcasts a federated value from the `tff.SERVER` to the `tff.CLIENTS`.

  Args:
    value: A value of a TFF federated type placed at the `tff.SERVER`, all
      members of which are equal (the `tff.FederatedType.all_equal` property of
      `value` is `True`).

  Returns:
    A representation of the result of broadcasting: a value of a TFF federated
    type placed at the `tff.CLIENTS`, all members of which are equal.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at the
      `tff.SERVER`.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_broadcast(value)


def federated_collect(value):
  """Returns a federated value from `tff.CLIENTS` as a `tff.SERVER` sequence.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.

  Returns:
    A stream of the same type as the member constituents of `value` placed at
    the `tff.SERVER`.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_collect(value)


def federated_map(mapping_fn, value):
  """Maps a federated value pointwise using a mapping function.

  The function `mapping_fn` is applied separately across the group of devices
  represented by the placement type of `value`. For example, if `value` has
  placement type `tff.CLIENTS`, then `mapping_fn` is applied to each client
  individually. In particular, this operation does not alter the placement of
  the federated value.

  Args:
    mapping_fn: A mapping function to apply pointwise to member constituents of
      `value`. The parameter of this function must be of the same type as the
      member constituents of `value`.
    value: A value of a TFF federated type (or a value that can be implicitly
      converted into a TFF federated type, e.g., by zipping) placed at
      `tff.CLIENTS` or `tff.SERVER`.

  Returns:
    A federated value with the same placement as `value` that represents the
    result of `mapping_fn` on the member constituent of `arg`.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_map(mapping_fn, value)


def federated_reduce(value, zero, op):
  """Reduces `value` from `tff.CLIENTS` to `tff.SERVER` using a reduction `op`.

  This method reduces a set of member constituents of a `value` of federated
  type `T@CLIENTS` for some `T`, using a given `zero` in the algebra (i.e., the
  result of reducing an empty set) of some type `U`, and a reduction operator
  `op` with type signature `(<U,T> -> U)` that incorporates a single `T`-typed
  member constituent of `value` into the `U`-typed result of partial reduction.
  In the special case of `T` equal to `U`, this corresponds to the classical
  notion of reduction of a set using a commutative associative binary operator.
  The generalized reduction (with `T` not equal to `U`) requires that repeated
  application of `op` to reduce a set of `T` always yields the same `U`-typed
  result, regardless of the order in which elements of `T` are processed in the
  course of the reduction.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.
    zero: The result of reducing a value with no constituents.
    op: An operator with type signature `(<U,T> -> U)`, where `T` is the type of
      the constituents of `value` and `U` is the type of `zero` to be used in
      performing the reduction.

  Returns:
    A representation on the `tff.SERVER` of the result of reducing the set of
    all member constituents of `value` using the operator `op` into a single
    item.

  Raises:
    TypeError: if the arguments are not of the types specified above.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_reduce(value, zero, op)


def federated_sum(value):
  """Computes a sum at `tff.SERVER` of a `value` placed on the `tff.CLIENTS`.

  To sum integer values with stronger privacy properties, consider using
  `tff.secure_sum`.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.

  Returns:
    A representation of the sum of the member constituents of `value` placed
    on the `tff.SERVER`.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_sum(value)


def federated_value(value, placement):
  """Returns a federated value at `placement`, with `value` as the constituent.

  Args:
    value: A value of a non-federated TFF type to be placed.
    placement: The desired result placement (either `tff.SERVER` or
      `tff.CLIENTS`).

  Returns:
    A federated value with the given placement `placement`, and the member
    constituent `value` equal at all locations.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_value(value, placement)


def federated_zip(value):
  """Converts an N-tuple of federated values into a federated N-tuple value.

  Args:
    value: A value of a TFF named tuple type, the elements of which are
      federated values with the same placement.

  Returns:
    A federated value placed at the same location as the members of `value`, in
    which every member component is a named tuple that consists of the
    corresponding member components of the elements of `value`.

  Raises:
    TypeError: if the argument is not a named tuple of federated values with the
      same placement.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.federated_zip(value)


def secure_sum(value, bitwidth):
  """Computes a sum at `tff.SERVER` of a `value` placed on the `tff.CLIENTS`.

  This function computes a sum such that it should not be possible for the
  server to learn any clients individual value. The specific algorithm and
  mechanism used to compute the secure sum may vary depending on the target
  runtime environment the computation is compiled for or executed on. See
  https://research.google/pubs/pub47246/ for more information.

  Not all executors support `tff.secure_sum()`; consult the documentation for
  the specific executor or executor stack you plan on using for the specific of
  how it's handled by that executor.

  TODO(b/148147384): Describe the semantics of secure sum intrinsic.

  Example:

  ```python
  value = tff.federated_value(1, tff.CLIENTS)
  result = tff.secure_sum(value, 2)

  value = tff.federated_value([1, 1], tff.CLIENTS)
  result = tff.secure_sum(value, [2, 4])

  value = tff.federated_value([1, [1, 1]], tff.CLIENTS)
  result = tff.secure_sum(value, [2, [4, 8]])
  ```

  NOTE: To sum non-integer values or to sum integers with fewer constraints and
  weaker privacy properties, consider using `federated_sum`.

  Args:
    value: A value of a TFF federated type placed at the `tff.CLIENTS`.
    bitwidth: An integer or nested structure of integers.

  Returns:
    A representation of the sum of the member constituents of `value` placed
    on the `tff.SERVER`.

  Raises:
    TypeError: if the argument is not a federated TFF value placed at
      `tff.CLIENTS`.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.secure_sum(value, bitwidth)


def sequence_map(mapping_fn, value):
  """Maps a TFF sequence `value` pointwise using a given function `mapping_fn`.

  This function supports two modes of usage:

  * When applied to a non-federated sequence, it maps individual elements of
    the sequence pointwise. If the supplied `mapping_fn` is of type `T->U` and
    the sequence `value` is of type `T*` (a sequence of `T`-typed elements),
    the result is a sequence of type `U*` (a sequence of `U`-typed elements),
    with each element of the input sequence individually mapped by `mapping_fn`.
    In this mode of usage, `sequence_map` behaves like a compuatation with type
    signature `<T->U,T*> -> U*`.

  * When applied to a federated sequence, `sequence_map` behaves as if it were
    individually applied to each member constituent. In this mode of usage, one
    can think of `sequence_map` as a specialized variant of `federated_map` that
    is designed to work with sequences and allows one to
    specify a `mapping_fn` that operates at the level of individual elements.
    Indeed, under the hood, when `sequence_map` is invoked on a federated type,
    it injects `federated_map`, thus
    emitting expressions like
    `federated_map(a -> sequence_map(mapping_fn, x), value)`.

  Args:
    mapping_fn: A mapping function to apply pointwise to elements of `value`.
    value: A value of a TFF type that is either a sequence, or a federated
      sequence.

  Returns:
    A sequence with the result of applying `mapping_fn` pointwise to each
    element of `value`, or if `value` was federated, a federated sequence
    with the result of invoking `sequence_map` on member sequences locally
    and independently at each location.

  Raises:
    TypeError: If the arguments are not of the appropriate types.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.sequence_map(mapping_fn, value)


def sequence_reduce(value, zero, op):
  """Reduces a TFF sequence `value` given a `zero` and reduction operator `op`.

  This method reduces a set of elements of a TFF sequence `value`, using a given
  `zero` in the algebra (i.e., the result of reducing an empty sequence) of some
  type `U`, and a reduction operator `op` with type signature `(<U,T> -> U)`
  that incorporates a single `T`-typed element of `value` into the `U`-typed
  result of partial reduction. In the special case of `T` equal to `U`, this
  corresponds to the classical notion of reduction of a set using a commutative
  associative binary operator. The generalized reduction (with `T` not equal to
  `U`) requires that repeated application of `op` to reduce a set of `T` always
  yields the same `U`-typed result, regardless of the order in which elements
  of `T` are processed in the course of the reduction.

  One can also invoke `sequence_reduce` on a federated sequence, in which case
  the reductions are performed pointwise; under the hood, we construct an
  expression  of the form
  `federated_map(a -> sequence_reduce(x, zero, op), value)`. See also the
  discussion on `sequence_map`.

  Args:
    value: A value that is either a TFF sequence, or a federated sequence.
    zero: The result of reducing a sequence with no elements.
    op: An operator with type signature `(<U,T> -> U)`, where `T` is the type of
      the elements of the sequence, and `U` is the type of `zero` to be used in
      performing the reduction.

  Returns:
    The `U`-typed result of reducing elements in the sequence, or if the `value`
    is federated, a federated `U` that represents the result of locally
    reducing each member constituent of `value`.

  Raises:
    TypeError: If the arguments are not of the types specified above.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.sequence_reduce(value, zero, op)


def sequence_sum(value):
  """Computes a sum of elements in a sequence.

  Args:
    value: A value of a TFF type that is either a sequence, or a federated
      sequence.

  Returns:
    The sum of elements in the sequence. If the argument `value` is of a
    federated type, the result is also of a federated type, with the sum
    computed locally and independently at each location (see also a discussion
    on `sequence_map` and `sequence_reduce`).

  Raises:
    TypeError: If the arguments are of wrong or unsupported types.
  """
  factory = intrinsic_factory.IntrinsicFactory(context_stack_impl.context_stack)
  return factory.sequence_sum(value)
