import ivy


def dot(inputs, axes, normalize):
    inputa = ivy.astype(ivy.copy_array(inputs[0]), ivy.float32)
    inputb = ivy.astype(ivy.copy_array(inputs[1]), ivy.float32)
    if normalize:
        return ivy.vecdot(
            ivy.vector_norm(inputa, axes), ivy.vector_norm(inputb, axes), axes
        )
    return ivy.vecdot(inputa, inputb, axes)


dot.supported_dtypes = {
    "numpy": (),
    "tensorflow": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ),
    "torch": (),
    "jax": (),
}
