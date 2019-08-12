# Macros for building CM code.

def cm_is_configured():
    """Returns true if CUDA was enabled during the configure process."""
    return %{cm_is_configured}

def if_cm_is_configured(x):
    """Tests if the CUDA was enabled during the configure process.

    Unlike if_cuda(), this does not require that we are building with
    --config=cuda. Used to allow non-CUDA code to depend on CUDA libraries.
    """
    if cm_is_configured():
        return x
    return []
