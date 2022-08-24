import numpy as np
from hypothesis import given, strategies as st, settings

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf


@st.composite
def _dot_helper(draw):
    shape = draw(helpers.get_shape(min_num_dims=2))
    axis = draw(helpers.get_axis(shape=shape, force_int=True))

    dtype = draw(st.sampled_from(ivy_tf.valid_dtypes))

    array1 = draw(helpers.array_values(shape=shape, dtype=dtype))
    array2 = draw(helpers.array_values(shape=shape, dtype=dtype))

    array1 = np.asarray(array1, dtype=dtype)
    array2 = np.asarray(array2, dtype=dtype)

    return [dtype, dtype], [array1, array2], axis


@settings(max_examples=100)
@given(
    dtypes_arrays_axes=_dot_helper(),
    normalize=st.booleans(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.dot"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_dot(
    *, dtypes_arrays_axes, normalize, as_variable, num_positional_args, native_array, fw
):
    dtypes, arrays, axes = dtypes_arrays_axes

    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.layers.dot",
        inputs=arrays,
        axes=axes,
        normalize=False,
    )
