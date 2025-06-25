# priority_logic.py

# A list of vehicles that are considered 'heavy' and will trigger slope priority.
heavy_vehicles = ["truck", "bus"]

def get_signal_with_slope_priority(
    label_left, near_left, approach_left,
    label_right, near_right, approach_right,
    left_orientation, right_orientation
):
    """
    Determines signal state for a sloped intersection.
    It gives priority to heavy vehicles and then falls back to normal traffic rules.
    """
    uphill_data, downhill_data = None, None

    # Assign camera data based on user-configured orientation
    if left_orientation == 'uphill':
        uphill_data = (label_left, near_left)
    elif left_orientation == 'downhill':
        downhill_data = (label_left, near_left)

    if right_orientation == 'uphill':
        uphill_data = (label_right, near_right)
    elif right_orientation == 'downhill':
        downhill_data = (label_right, near_right)

    # Check for heavy vehicles on the slope
    uphill_heavy = uphill_data and uphill_data[0] in heavy_vehicles and uphill_data[1]
    downhill_heavy = downhill_data and downhill_data[0] in heavy_vehicles and downhill_data[1]

    # --- Case 1: Slope Priority Logic ---
    if uphill_heavy and downhill_heavy:
        return 'red', "DANGER: Heavy vehicles on both uphill and downhill slopes!"
    if uphill_heavy:
        return 'red', f"PRIORITY: Heavy load vehicle uphill."
    if downhill_heavy:
        return 'red', f"PRIORITY: Heavy load vehicle downhill."

    # --- Case 2: Fallback to Normal Logic if no priority is triggered ---
    return get_signal_no_slope(
        label_left, near_left, approach_left,
        label_right, near_right, approach_right
    )

def get_signal_no_slope(
    label_left, near_left, approach_left,
    label_right, near_right, approach_right
):
    """
    Determines signal state for a standard, flat intersection.
    """
    if near_left and near_right:
        return 'red', "Conflict: Vehicles near in both directions."
    if (near_left and approach_right) or (near_right and approach_left):
        return 'yellow', "Caution: Vehicles approaching from both sides."
    if near_left or near_right:
        return 'green', "Path clear for one direction."
        
    return 'green', "All clear."
