def get_weightclass(weight: str) -> str:
    """
    Converts a given weight in pounds to its corresponding weight class.

    Args:
        weight (str): Weight in pounds (e.g., '154 lbs.').

    Returns:
        str: The corresponding weight class (e.g., 'Featherweight').
    """
    weight = int(weight.rstrip(" lbs."))
    weight_classes = {
        134: "Flyweight",
        144: "Bantamweight",
        154: "Featherweight",
        169: "Lightweight",
        184: "Welterweight",
        204: "Middleweight",
        224: "Lightheavyweight",
        float('inf'): "Heavyweight"
    }
    for upper_limit, weight_class in weight_classes.items():
        if weight < upper_limit:
            return weight_class


def inch_to_cm(height: str) -> float:
    """
    Converts a given height in feet and inches to centimeters.

    Args:
        height (str): Height in feet and inches (e.g., "5'8"") or just inches (e.g., '68"').

    Returns:
        float: The height converted to centimeters, rounded to one decimal place.
    """
    if "'" in height:
        feet, inches = height.split("'")
        total_inches = (int(feet) * 12) + int(inches.strip('"'))
        cm = total_inches * 2.54
        return round(cm, 1)
    
    elif '"' in height:
        inches = int(height.strip('"'))
        return round(inches * 2.54, 1)