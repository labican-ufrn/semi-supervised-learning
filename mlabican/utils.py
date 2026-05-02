def has_method(obj: object, method_name: str):
    """
    Check if an object has a callable method with the given name.

    Args:
        obj (object): Callable object.
        method_name (str): The name of the method that you want to check.

    Returns:
        bool: True if the object has the method that match with the name
            False, otherwise.
    """
    attribute = getattr(obj, method_name, None)

    return callable(attribute)
