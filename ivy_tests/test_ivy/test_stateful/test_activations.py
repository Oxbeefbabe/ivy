"""Collection of tests for unified neural network activations."""

# global
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# GELU
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy.valid_numeric_dtypes),
    approximate=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="GELU.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="GELU._forward"),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={"approximate": approximate},
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"x": np.asarray(x, dtype=input_dtype)},
        fw=fw,
        class_name="GELU",
        method_name="_forward",
    )


# GEGLU
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.valid_numeric_dtypes, num_arrays=2
    ),
    approximate=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="GEGLU.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="GEGLU._forward"),
)
def test_geglu(
    *,
    dtype_and_x,
    approximate,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtypes, x = dtype_and_x
    inputs = np.asarray(x[0], dtype=input_dtypes[0])
    gates = np.asarray(x[1], dtype=input_dtypes[1])
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={"approximate": approximate},
        input_dtypes_method=input_dtypes,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": inputs, "gates": gates},
        fw=fw,
        class_name="GEGLU",
        method_name="_forward",
    )
