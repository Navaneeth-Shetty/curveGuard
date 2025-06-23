def estimate_road_condition(temp, humidity):
    if humidity > 80 and temp < 20:
        return "wet"
    return "dry"