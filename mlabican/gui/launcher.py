def launch_parameter_gui() -> dict | None:
    """Opens the ParameterForm GUI and returns the validated parameters."""
    from mlabican.gui.parameter_form import ParameterForm

    app = ParameterForm()
    app.parameters = None
    app.mainloop()

    return app.parameters
