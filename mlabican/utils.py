def has_method(obj, method_name):
    """
    Check if an object has a callable method with the given name.

    At
    """
    attribute = getattr(obj, method_name, None)

    return callable(attribute)
