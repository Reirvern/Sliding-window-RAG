def get_interface(interface_type: str):
    if interface_type == "cli":
        return CLIInterface()
    elif interface_type == "web":
        return WebServer()