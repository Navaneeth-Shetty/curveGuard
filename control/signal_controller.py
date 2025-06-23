def set_light(decision):
    """
    Controls signal and returns a decision string
    Accepted decision values (input): "right", "left", "both_red", "both_green"
    Returns clean string used for logging and ML model: 
    - "RIGHT GREEN | LEFT RED"
    - "LEFT GREEN | RIGHT RED"
    - "BOTH RED"
    - "BOTH GREEN"
    """
    if decision == "right":
        print("RIGHT GREEN | LEFT RED")
        return "RIGHT GREEN | LEFT RED"
    elif decision == "left":
        print("LEFT GREEN | RIGHT RED")
        return "LEFT GREEN | RIGHT RED"
    elif decision == "both_red":
        print("BOTH RED")
        return "BOTH RED"
    else:
        print("BOTH GREEN")
        return "BOTH GREEN"
