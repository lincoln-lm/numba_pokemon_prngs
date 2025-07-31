"""Monkey patching"""

from .options import USE_NUMBA


def patch_numba_8e2063b():
    if USE_NUMBA:
        from numba.core.types.npytypes import Record, Array, NestedArray

        old_record_init = Record.__init__

        def new_record_init(self, *args, **kwargs):
            old_record_init(self, *args, **kwargs)
            self.bitwidth = self.dtype.itemsize * 8

        Record.__init__ = new_record_init

        def new_array_init(
            self, dtype, ndim, layout, readonly=False, name=None, aligned=True
        ):
            if readonly:
                self.mutable = False
            if not aligned or (isinstance(dtype, Record) and not dtype.aligned):
                self.aligned = False
            if isinstance(dtype, NestedArray):
                ndim += dtype.ndim
                dtype = dtype.dtype
            if name is None:
                type_name = "array"
                if not self.mutable:
                    type_name = "readonly " + type_name
                if not self.aligned:
                    type_name = "unaligned " + type_name
                name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
            super(Array, self).__init__(dtype, ndim, layout, name=name)

        Array.__init__ = new_array_init

        def new_nestedarray_init(self, dtype, shape):
            if isinstance(dtype, NestedArray):
                tmp = Array(dtype.dtype, dtype.ndim, "C")
                shape += dtype.shape
                dtype = tmp.dtype
            assert dtype.bitwidth % 8 == 0, "Dtype bitwidth must be a multiple of bytes"
            self._shape = shape
            name = "nestedarray(%s, %s)" % (dtype, shape)
            ndim = len(shape)
            super(NestedArray, self).__init__(dtype, ndim, "C", name=name)

        NestedArray.__init__ = new_nestedarray_init
