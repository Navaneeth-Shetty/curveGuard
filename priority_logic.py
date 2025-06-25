# priority_logic.py

# A list of vehicles that are considered 'heavy' and will trigger slope priority.
# You can easily add or remove vehicle types here (e.g., "tractor").
heavy_vehicles = ["truck", "bus"]

def determine_signal_state(
    label_uphill, near_uphill, approach_uphill,
    label_downhill, near_downhill, approach_downhill
):
    """
    Determines the correct signal state based on all traffic conditions,
    giving precedence to slope priority.

    This function covers all logic cases:
    1.  Slope priority for heavy vehicles.
    2.  Normal traffic conflict resolution.

    Args:
        label_uphill (str): Vehicle label on the uphill path (right camera).
        near_uphill (bool): Is the uphill vehicle near?
        approach_uphill (bool): Is the uphill vehicle approaching?
        label_downhill (str): Vehicle label on the downhill path (left camera).
        near_downhill (bool): Is the downhill vehicle near?
        approach_downhill (bool): Is the downhill vehicle approaching?

    Returns:
        tuple: (str: new_signal, str: reason)
               - new_signal: 'green', 'yellow', or 'red'
               - reason: A string explaining the decision for logging/display.
    """
    # --- Case 1: Slope Priority for Heavy Vehicles (Highest Precedence) ---
    # A heavy vehicle going UPHILL gets priority.
    if label_uphill in heavy_vehicles and near_uphill:
        return 'red', f"Priority for uphill {label_uphill}."
    
    # A heavy vehicle going DOWNHILL gets priority for safety.
    if label_downhill in heavy_vehicles and near_downhill:
        return 'red', f"Priority for downhill {label_downhill}."

    # --- Case 2: Normal Traffic Logic (if no slope priority) ---
    # If vehicles are near on both sides, it's a conflict -> RED
    if near_downhill and near_uphill:
        return 'red', "Conflict: Vehicles near in both directions."

    # If one vehicle is near and the other is approaching -> YELLOW (caution)
    if (near_downhill and approach_uphill) or (near_uphill and approach_downhill):
        return 'yellow', "Caution: Vehicles approaching from both sides."

    # If a vehicle is detected on either side but it's not a conflict -> GREEN
    if near_downhill or near_uphill:
        return 'green', "Path clear for one direction."

    # Default case: No vehicles are near or approaching -> GREEN
    return 'green', "All clear."

