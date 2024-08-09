"""Functions and classes to do with numba compilation"""

from typing import get_type_hints
import numpy as np
from .options import USE_NUMBA

if USE_NUMBA:
    import numba
    from numba.core.imputils import lower_builtin
    from numba.core.typing.templates import AbstractTemplate, signature, infer_global
    from numba.types import string as numba_string

    # pylint: disable=unused-import,no-name-in-module
    from numba.typed import List as TypedList
    from numba.typed import Dict as TypedDict_
    from numba.types import ListType as TypedListType

    # pylint: enable=unused-import,no-name-in-module

    def convert_to_numba(input_type):
        """Convert to numba type"""
        return (
            input_type
            if isinstance(input_type, numba.types.Type)
            else (
                numba.void
                if input_type == np.void
                else (
                    numba_string
                    if input_type == np.str_
                    else (
                        input_type.class_type.instance_type
                        if hasattr(input_type, "class_type")
                        else numba.from_dtype(input_type)
                    )
                )
            )
        )

    def typed_dict_constructor(key_type, value_type):
        """Construct TypedDict"""
        return TypedDict_.empty(
            key_type=convert_to_numba(key_type), value_type=convert_to_numba(value_type)
        )

    TypedDict = typed_dict_constructor

    def optional_ir_function(function, sig):
        """Optional custom IR function to be used in place of `function`"""

        @infer_global(function)
        class _CustomTemplate(AbstractTemplate):
            """Custom Typing Template"""

            def generic(self, args, _kws):
                """Generic function signature match"""
                return signature(sig.return_type, *args)

        return lower_builtin(function, *sig.args)

    def unituple_type(np_dtype, count):
        """Corresponding type for a unituple of np_dtype"""
        return numba.types.UniTuple(numba.from_dtype(np_dtype), count)

    def array_type(np_dtype, ndim: int = 1, layout: str = "C", readonly: bool = False):
        """Corresponding type for an array of np_dtype"""
        return numba.types.Array(
            numba.from_dtype(np_dtype), ndim, layout, readonly=readonly
        )

    def return_type(output_type, input_types):
        """Function that takes inputs and returns the return type based on the output type"""
        if not hasattr(input_types, "__iter__"):
            raise TypeError(f"{input_types} is not a iter of inputs")
        output_type = convert_to_numba(output_type)
        return output_type(
            *(convert_to_numba(input_type) for input_type in input_types)
        )

    def convert_spec(spec):
        """Convert a spec containing numpy types to use numba types"""
        for attr, py_type in spec.copy().items():
            spec[attr] = convert_to_numba(py_type)
        return spec

    def convert_spec_cls(spec, cls):
        """Convert a spec built from type hints containing numpy types to use numba types"""
        spec = spec or {}
        for attr, py_type in get_type_hints(cls).items():
            spec[attr] = convert_to_numba(py_type)
        return spec

else:
    # pylint disable=unused-argument
    def typed_dict_constructor(key_type, value_type):
        """Construct TypedDict"""
        return {}

    TypedList = list
    TypedListType = list
    TypedDict = typed_dict_constructor

    def convert_to_numba(input_type):
        """Convert to numba type"""
        return input_type

    def optional_ir_function(function, sig):
        """Optional custom IR function to be used in place of `function`"""
        raise NotImplementedError("IR Functions are not implemented outside of numba")

    def unituple_type(np_dtype, count):
        """Corresponding type for a unituple of np_dtype"""
        return tuple

    def array_type(np_dtype, ndim: int = 1, layout: str = "C", readonly: bool = False):
        """Corresponding type for an array of np_dtype"""
        return np.ndarray

    def return_type(output_type, input_types):
        """Function that takes inputs and returns the return type based on the output type"""
        return lambda *args: output_type

    # pylint enable=unused-argument


def optional_jitclass(cls_or_spec=None, spec=None):
    """Optional numba.experimental.jitclass"""
    if USE_NUMBA:
        if (
            cls_or_spec is not None
            and spec is None
            and not isinstance(cls_or_spec, type)
        ):
            cls_or_spec = convert_spec(cls_or_spec)
        elif spec is not None:
            spec = convert_spec(spec)

        if cls_or_spec is None:

            def wrap(cls):
                spec = convert_spec_cls(None, cls)
                return numba.experimental.jitclass(cls_or_spec=cls, spec=spec)

            return wrap

        if isinstance(cls_or_spec, type):
            spec = convert_spec_cls(spec, cls_or_spec)
        return numba.experimental.jitclass(cls_or_spec=cls_or_spec, spec=spec)
    else:
        if not isinstance(cls_or_spec, type):

            def wrap(cls):
                return cls

            return wrap
        return cls_or_spec


def optional_njit(signature_or_function=None, locals=None, **options):
    """Optional numba.njit"""
    if locals is None:
        locals = {}
    if USE_NUMBA:
        return numba.njit(
            signature_or_function=signature_or_function,
            locals=convert_spec(locals),
            **options,
        )

    # TODO: support @optional_njit w/o arguments provided

    def wrap(func):
        return func

    return wrap
