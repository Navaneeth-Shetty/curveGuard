# priority_logic.py

# A list of vehicles that are considered 'heavy' and will trigger slope priority.
heavy_vehicles = ["truck", "bus"]

def get_signal_with_slope_priority(
    label_left, near_left, approach_left,
    label_right, near_right, approach_right,
    left_orientation, right_orientation
):
    """
    Determines signal state for a sloped intersection based on new priority rules.

    New Logic:
    1. A lone heavy vehicle gets a GREEN light.
    2. If a heavy vehicle is on one side and ANY vehicle appears on the other,
       the signal turns RED to give the heavy vehicle priority.
    3. If heavy vehicles are on both sides, the signal turns RED for safety.
    """
    # --- Data Preparation: Map left/right cameras to uphill/downhill concepts ---
    uphill_label, downhill_label = "none", "none"
    near_uphill, near_downhill = False, False

    if left_orientation == 'uphill':
        uphill_label, near_uphill = label_left, near_left
    elif left_orientation == 'downhill':
        downhill_label, near_downhill = label_left, near_left

    if right_orientation == 'uphill':
        uphill_label, near_uphill = label_right, near_right
    elif right_orientation == 'downhill':
        downhill_label, near_downhill = label_right, near_right

    # Check if a heavy vehicle is present on either slope
    is_heavy_uphill = uphill_label in heavy_vehicles and near_uphill
    is_heavy_downhill = downhill_label in heavy_vehicles and near_downhill

    # --- New Priority Logic ---

    # Case 1: Highest Conflict -> Heavy vehicles on BOTH sides.
    if is_heavy_uphill and is_heavy_downhill:
        return 'red', "DANGER: Heavy vehicles on both uphill and downhill slopes!"

    # Case 2: Priority Conflict -> Heavy vehicle on one side, ANY vehicle on the other.
    # The signal must be RED to let the heavy vehicle pass safely.
    if is_heavy_uphill and near_downhill:
        return 'red', "PRIORITY: Giving way to heavy vehicle uphill."
    if is_heavy_downhill and near_uphill:
        return 'red', "PRIORITY: Giving way to heavy vehicle downhill."

    # Case 3: Lone Heavy Vehicle -> The path is clear for it.
    # This is only triggered if the other side is empty.
    if is_heavy_uphill:
        return 'green', "Path clear for heavy vehicle on uphill."
    if is_heavy_downhill:
        return 'green', "Path clear for heavy vehicle on downhill."

    # Case 4: No Heavy Vehicles Involved -> Fallback to normal traffic rules.
    # This runs if there are no heavy vehicles, or if they are not near.
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
    (This function is unchanged and acts as the fallback.)
    """
    if near_left and near_right:
        return 'red', "Conflict: Vehicles near in both directions."
    if (near_left and approach_right) or (near_right and approach_left):
        return 'yellow', "Caution: Vehicles approaching from both sides."
    if near_left or near_right:
        return 'green', "Path clear for one direction."
        
    return 'green', "All clear."
