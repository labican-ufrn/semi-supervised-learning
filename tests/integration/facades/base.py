class BaseFacade:
    """Base facade for BDD tests."""

    def __init__(self, context) -> None:
        self.context = context
