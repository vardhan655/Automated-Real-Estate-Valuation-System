"""
Currency formatting utilities for Indian Real Estate (INR).

Supports:
- Indian numbering system with commas: ₹75,50,000 / ₹1,85,00,000
- Short representation: ₹75.50 L / ₹1.85 Cr
- Compact representation for metrics and plots
"""

from typing import Union


def format_inr(amount: Union[float, int], include_symbol: bool = True) -> str:
    """
    Format a number in the Indian numbering system.

    In the Indian system:
    - Last 3 digits are grouped together.
    - Subsequent digits are grouped in pairs of 2.
    - Example: 7550000 -> 75,50,000; 18500000 -> 1,85,00,000.

    Args:
        amount: Price or amount in Indian Rupees (INR).
        include_symbol: Whether to prepend the ₹ symbol.

    Returns:
        Formatted string like "₹75,50,000" or "75,50,000".
    """
    if amount is None or (isinstance(amount, float) and (amount != amount)):  # NaN check
        return "N/A"

    is_negative = amount < 0
    amount = abs(round(float(amount)))
    s = str(int(amount))

    if len(s) <= 3:
        formatted = s
    else:
        # Last 3 digits
        last_three = s[-3:]
        # Preceding digits grouped in 2s from right to left
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        groups.reverse()
        formatted = ",".join(groups) + "," + last_three

    prefix = "-" if is_negative else ""
    sym = "₹" if include_symbol else ""
    return f"{prefix}{sym}{formatted}"


def format_inr_short(amount: Union[float, int], include_symbol: bool = True) -> str:
    """
    Format amount in Indian Lakhs (L) and Crores (Cr).

    Examples:
        75,00,000 -> ₹75.00 L
        1,50,00,000 -> ₹1.50 Cr
        50,000 -> ₹50,000

    Args:
        amount: Price or amount in INR.
        include_symbol: Whether to prepend ₹ symbol.

    Returns:
        Short string representation.
    """
    if amount is None or (isinstance(amount, float) and (amount != amount)):
        return "N/A"

    is_negative = amount < 0
    val = abs(float(amount))
    sym = "₹" if include_symbol else ""
    prefix = "-" if is_negative else ""

    if val >= 10_000_000:  # 1 Crore = 10 Million
        cr = val / 10_000_000
        return f"{prefix}{sym}{cr:.2f} Cr"
    elif val >= 100_000:  # 1 Lakh = 100 Thousand
        lakh = val / 100_000
        return f"{prefix}{sym}{lakh:.2f} L"
    else:
        return format_inr(amount, include_symbol=include_symbol)
