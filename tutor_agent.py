"""Adaptive Math Tutor - Core Agent Tools.

This module provides deterministic calculation tools and diagnostic schemas for the
math tutor agent to ensure ground-truth mathematical accuracy and structured pedagogical feedback.
"""

import os
import json
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dataclasses import dataclass, field


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
    print(f"\n[TOOL RUNNING] calculate_arithmetic({operand_a}, {operand_b}, '{operator}')")
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


class DiagnosedError(str, Enum):
    """Fine-grained cognitive error classifications for early elementary math."""

    NONE = "none"
    OFF_BY_ONE = "off_by_one"
    OPERATOR_CONFUSION = "operator_confusion"
    REGROUPING_FAILURE = "regrouping_failure"
    REVERSAL_ERROR = "reversal_error"
    RANDOM_GUESS = "random_guess"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_lower = value.lower().strip().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return cls.OTHER


class RecommendedNextAction(str, Enum):
    """Recommended pedagogical next actions for the tutoring session."""

    PROCEED = "proceed"
    PROVIDE_HINT = "provide_hint"
    REQUEST_RETRY = "request_retry"
    BREAK_DOWN_STEP = "break_down_step"
    REVIEW_CONCEPT = "review_concept"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_lower = value.lower().strip()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return cls.OTHER


class DiagnosticOutput(BaseModel):
    """Structured diagnostic evaluation representing the math tutor agent's assessment."""

    is_correct: bool = Field(
        ...,
        description="Indicates whether the student's solution or step is mathematically correct.",
    )
    diagnosed_error: DiagnosedError = Field(
        ...,
        description="The category of error identified in the student's response ('none' if correct).",
    )
    pedagogical_clue: str = Field(
        ...,
        description="A supportive, Socratic clue or hint to help the student identify and resolve their error without directly revealing the final answer.",
    )
    recommended_next_action: RecommendedNextAction = Field(
        ...,
        description="The recommended pedagogical next action to steer the tutoring flow (e.g., proceed, provide_hint, request_retry).",
    )


class ScaffoldLevel(str, Enum):
    """Scaffolding tiers for adaptive problem delivery."""

    NONE = "none"  # Standard abstract equation (e.g., "7 + 4 = ?")
    HINT = "hint"  # Equation + strategic tip
    VISUAL = "visual"  # Concrete story problem with countable items/emojis


class GeneratedProblem(BaseModel):
    """Schema for an adaptively generated math problem."""

    equation: str = Field(..., description="The mathematical expression to solve, e.g. '7 + 4'")
    story_context: str = Field(..., description="A friendly 1st-grade narrative framing the problem.")
    scaffold_level: ScaffoldLevel = Field(..., description="The tier of scaffolding applied.")
    target_skill: str = Field(..., description="The specific arithmetic concept being reinforced.")


@dataclass
class StudentSessionState:
    """Tracks running progress and cognitive gaps across the session."""

    student_name: str
    target_skill: str = "addition_within_20"
    mastery_score: int = 50  # Scale 0 to 100
    recent_errors: list[DiagnosedError] = field(default_factory=list)

    def record_attempt(self, diagnosis: DiagnosticOutput) -> None:
        """Updates internal mastery heuristics based on the diagnostic assessment."""
        if diagnosis.is_correct:
            self.mastery_score = min(100, self.mastery_score + 15)
        else:
            self.recent_errors.append(diagnosis.diagnosed_error)
            self.mastery_score = max(0, self.mastery_score - 10)


def evaluate_attempt(problem: str, student_answer: int) -> DiagnosticOutput:
    """Invokes the diagnostic agent, executing automatic tool calling and returning structured feedback."""
    client = genai.Client()

    system_instruction = f"""
        You are the diagnostic engine for an adaptive 1st-grade math tutor.
        Target Audience: 6-7 year old students learning addition and subtraction.

        Operational Rules:
        1. Ground Truth First: You are strictly forbidden from doing mental arithmetic. You MUST call the calculate_arithmetic tool to compute the true mathematical result.
        2. Error Classification: After receiving the tool calculation, compare it with the student's answer and identify the misconception.
        3. Response Format: Provide your final evaluation strictly as a valid JSON object matching this schema:
        {json.dumps(DiagnosticOutput.model_json_schema(), indent=2)}

        Do NOT wrap the JSON in markdown fences (no ```json). Output ONLY the raw JSON object.
        """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        tools=[calculate_arithmetic],
        # thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disables reasoning delay
    )

    prompt = f"Problem: {problem}\nStudent Answer: {student_answer}"

    print("[DEBUG] Sending request to Gemini...")
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=config,
    )
    print("[DEBUG] Response received from Gemini!")

    clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    return DiagnosticOutput.model_validate_json(clean_text)

def generate_next_problem(state: StudentSessionState) -> GeneratedProblem:
    """Invokes the pedagogical generator to create a problem tailored to the student's current state."""
    client = genai.Client()

    system_instruction = f"""
        You are the Pedagogical Content Generator for a 1st-grade adaptive math tutor.
        Target Audience: 6-7 year old students.

        Your Task:
        Generate ONE targeted math problem calibrated to the student's current mastery level and recent misconceptions.

        Pedagogical Rules:
        1. If the student has recent OFF_BY_ONE or REGROUPING_FAILURE errors, set scaffold_level to 'visual' and use concrete objects (frogs, blocks, apples) to prevent counting drops past 10.
        2. If mastery_score >= 75, set scaffold_level to 'none' and test clean abstract equations.
        3. Keep addition sums between 5 and 20. Do NOT exceed 20.
        4. Output strictly a JSON object conforming to this schema:
        {json.dumps(GeneratedProblem.model_json_schema(), indent=2)}

        Do NOT wrap the output in markdown fences. Output ONLY the raw JSON object.
        """

    prompt = f"""
        Student Name: {state.student_name}
        Target Skill: {state.target_skill}
        Mastery Score: {state.mastery_score}/100
        Recent Misconceptions: {[e.value for e in state.recent_errors[-3:]]}

        Generate the next practice problem.
        """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
        ),
    )

    clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    return GeneratedProblem.model_validate_json(clean_text)

if __name__ == "__main__":
    # print("Testing calculate_arithmetic:")
    # print("12 + 4 =", calculate_arithmetic(12, 4, "+"))
    # print("12 / 4 =", calculate_arithmetic(12, 4, "/"))
    # print("12 / 0 =", calculate_arithmetic(12, 0, "/"))

    # print("\nTesting DiagnosticOutput:")
    # sample_correct = DiagnosticOutput(
    #     is_correct=True,
    #     diagnosed_error=DiagnosedError.NONE,
    #     pedagogical_clue="Excellent work calculating the sum correctly!",
    #     recommended_next_action=RecommendedNextAction.PROCEED,
    # )
    # print("Sample Correct:", sample_correct.model_dump())

    # sample_incorrect = DiagnosticOutput(
    #     is_correct=False,
    #     diagnosed_error="calculation_error",
    #     pedagogical_clue="Notice what happens when you add the digits in the tens column. Did you remember to carry the 1?",
    #     recommended_next_action="provide_hint",
    # )
    # print("Sample Incorrect:", sample_incorrect.model_dump())
    # print("--- Running Live Tutor Agent Test ---")
    
    # test_problem = "8 + 5"
    # test_answer = 12

    # print(f"Evaluating student attempt: {test_problem} = {test_answer}...")
    # diagnosis = evaluate_attempt(test_problem, test_answer)

    # print("\n--- Diagnostic Agent Result ---")
    # print(f"Is Correct:         {diagnosis.is_correct}")
    # print(f"Diagnosed Error:    {diagnosis.diagnosed_error.value}")
    # print(f"Pedagogical Clue:   {diagnosis.pedagogical_clue}")
    # print(f"Next Action:        {diagnosis.recommended_next_action.value}")
    print("=== Adaptive Math Tutor Session Initialized ===")
    state = StudentSessionState(student_name="Manas")

    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} (Mastery: {state.mastery_score}%) ---")
        problem = generate_next_problem(state)

        print(f"Skill Focus:  {problem.target_skill}")
        print(f"Scaffolding:  {problem.scaffold_level.value}")
        print(f"Story:        {problem.story_context}")
        print(f"Problem:      {problem.equation}")

        user_input = input("\nStudent's Answer (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break

        try:
            student_val = int(user_input)
        except ValueError:
            print("Please enter a valid whole number.")
            continue

        diagnosis = evaluate_attempt(problem.equation, student_val)
        state.record_attempt(diagnosis)

        print(f"\n[Evaluator] Correct? {diagnosis.is_correct}")
        print(f"[Evaluator] Diagnosed Gap: {diagnosis.diagnosed_error.value}")
        print(f"[Tutor Hint] {diagnosis.pedagogical_clue}")

    print("\nSession complete. Final Mastery Score:", state.mastery_score)