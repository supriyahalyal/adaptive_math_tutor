"""Adaptive Math Tutor - Core Agent Tools.

This module provides deterministic calculation tools for the math tutor agent
to ensure ground-truth mathematical accuracy during tutoring sessions.
"""

from typing import Any


def calculate_arithmetic(
    operand_a: int, operand_b: int, operator: str
) -> int | dict[str, Any]:
    """Computes exact arithmetic (+, -, *, /) and must be consulted to verify ground-truth math.

    This function serves as the authoritative, deterministic tool for verifying mathematical
    calculations within the tutoring agent. Language models can struggle with multi-digit
    arithmetic, edge cases, and hallucinations; hence, this function must be consulted
    whenever ground-truth math verification is needed.

    Parameters:
        operand_a (int): The first integer operand.
        operand_b (int): The second integer operand.
        operator (str): The arithmetic operation to execute. Supported operators are:
            - '+' : Exact addition
            - '-' : Exact subtraction
            - '*' : Exact multiplication
            - '/' : Exact integer division (when evenly divisible)

    Returns:
        int | dict[str, Any]:
            - Returns an int when the operation evaluates to an exact integer result.
            - Returns a dict with error or breakdown information when:
                * Division by zero is attempted (`{"error": "..."}`).
                * An unsupported operator is provided (`{"error": "..."}`).
                * Division does not divide evenly (`{"error": "...", "quotient": ..., "remainder": ...}`).

    Examples:
        >>> calculate_arithmetic(12, 4, "+")
        16
        >>> calculate_arithmetic(12, 4, "/")
        3
        >>> calculate_arithmetic(12, 0, "/")
        {'error': 'Division by zero is undefined.'}
        >>> calculate_arithmetic(13, 4, "/")
        {'error': 'Division is not exact: 13 is not evenly divisible by 4.', 'quotient': 3, 'remainder': 1}
    """
    match operator:
        case "+":
            return operand_a + operand_b
        case "-":
            return operand_a - operand_b
        case "*":
            return operand_a * operand_b
        case "/":
            if operand_b == 0:
                return {"error": "Division by zero is undefined."}
            if operand_a % operand_b != 0:
                return {
                    "error": f"Division is not exact: {operand_a} is not evenly divisible by {operand_b}.",
                    "quotient": operand_a // operand_b,
                    "remainder": operand_a % operand_b,
                }
            return operand_a // operand_b
        case _:
            return {
                "error": f"Unsupported operator '{operator}'. Supported operators are '+', '-', '*', '/'."
            }

if __name__ == "__main__":
    print(calculate_arithmetic(12, 4, "+"))
    print(calculate_arithmetic(12, 4, "-"))
    print(calculate_arithmetic(12, 4, "*"))
    print(calculate_arithmetic(12, 4, "/"))
    print(calculate_arithmetic(12, 0, "/"))
    print(calculate_arithmetic(13, 4, "/"))