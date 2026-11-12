"""Adaptive Math Tutor - Dual-Agent Implementation via LangChain.

This module coordinates a Diagnostic Agent (evaluating student responses with
deterministic tool verification) and a Content Generator (synthesizing targeted,
scaffolded practice problems) using LangChain abstractions.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import random
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# =====================================================================
# 1. Deterministic Tool
# =====================================================================
@tool
def calculate_arithmetic(operand_a: int, operand_b: int, operator: str) -> str:
    """Computes exact arithmetic (+, -, *, /) and must be consulted to verify ground-truth math.

    Args:
        operand_a: First integer operand.
        operand_b: Second integer operand.
        operator: Arithmetic operator ('+', '-', '*', or '/').
    """
    match operator:
        case "+":
            return str(operand_a + operand_b)
        case "-":
            return str(operand_a - operand_b)
        case "*":
            return str(operand_a * operand_b)
        case "/":
            if operand_b == 0:
                return "Error: Division by zero is undefined."
            if operand_a % operand_b != 0:
                return f"Error: {operand_a} is not evenly divisible by {operand_b}."
            return str(operand_a // operand_b)
        case _:
            return f"Error: Unsupported operator '{operator}'."


# =====================================================================
# 2. Schemas & Enums
# =====================================================================
class DiagnosedError(str, Enum):
    """Cognitive error classifications for early elementary arithmetic."""

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
    """Next pedagogical actions for the student session."""

    PROCEED = "proceed"
    PROVIDE_HINT = "provide_hint"
    REQUEST_RETRY = "request_retry"
    BREAK_DOWN_STEP = "break_down_step"
    REVIEW_CONCEPT = "review_concept"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_lower = value.lower().strip().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return cls.OTHER


class ScaffoldLevel(str, Enum):
    """Scaffolding tiers applied to generated problems."""

    NONE = "none"
    HINT = "hint"
    VISUAL = "visual"


class DiagnosticOutput(BaseModel):
    """Evaluation contract emitted by the Diagnostic Agent."""

    is_correct: bool = Field(
        ...,
        description="Whether the student's solution is mathematically correct.",
    )
    diagnosed_error: DiagnosedError = Field(
        ...,
        description="Identified error classification ('none' if correct).",
    )
    pedagogical_clue: str = Field(
        ...,
        description="Supportive, Socratic clue without directly stating the answer.",
    )
    recommended_next_action: RecommendedNextAction = Field(
        ...,
        description="Recommended next step for the tutoring flow.",
    )




class GeneratedProblem(BaseModel):
    """Problem contract emitted by the Curriculum Generator."""

    operand_a: int = Field(
        ...,
        description="The first whole number operand. Digits only.",
    )
    operand_b: int = Field(
        ...,
        description="The second whole number operand. Digits only.",
    )
    # Strictly lock allowed values in the JSON schema
    operator: Literal["+", "-"] = Field(
        "+",
        description="The arithmetic operation. MUST be strictly '+' or '-'.",
    )
    story_context: str = Field(
        ...,
        description="A 1st-grade friendly narrative setting up the equation.",
    )
    scaffold_level: ScaffoldLevel = Field(
        ...,
        description="The level of scaffolding applied.",
    )
    target_skill: str = Field(
        ...,
        description="The mathematical concept being exercised.",
    )
    visual_emoji: str = Field(
        "\U0001F34E",
        description="A single emoji matching the item counted in the story (e.g., \U0001F438, \U0001F680, \U0001F986, \U0001F36A).",
    )

    @field_validator("visual_emoji", mode="before")
    @classmethod
    def sanitize_emoji(cls, v: Any) -> str:
        """Ensures a single clean emoji is returned even if the SLM outputs text."""
        if isinstance(v, str):
            val = v.strip()
            if val and ord(val[0]) > 127:
                return val.split()[0]
        return "\U0001F34E"

    @field_validator("operator", mode="before")
    @classmethod
    def clean_operator_syntax(cls, v: Any) -> str:
        """Strips out accidental code syntax like +=, -=, or trailing spaces."""
        if isinstance(v, str):
            v_clean = v.strip()
            if "+" in v_clean:
                return "+"
            if "-" in v_clean:
                return "-"
        return "+"

    @property
    def equation(self) -> str:
        """Returns standard equation string for display and evaluation."""
        return f"{self.operand_a} {self.operator} {self.operand_b}"


# =====================================================================
# 3. Session State
# =====================================================================
@dataclass
class StudentSessionState:
    """Maintains student history, mastery scores, and completed equations across turns."""

    student_name: str
    target_skill: str = "addition_within_20"
    mastery_score: int = 50
    recent_errors: list[DiagnosedError] = field(default_factory=list)
    completed_equations: list[str] = field(default_factory=list)

    def record_attempt(self, equation: str, diagnosis: DiagnosticOutput) -> None:
        """Updates belief state and archives completed problems."""
        self.completed_equations.append(equation.replace(" ", "").replace("=?", "").replace("?", ""))
        
        if diagnosis.is_correct:
            self.mastery_score = min(100, self.mastery_score + 15)
            if self.recent_errors:
                self.recent_errors.pop(0)
        else:
            self.recent_errors.append(diagnosis.diagnosed_error)
            self.mastery_score = max(0, self.mastery_score - 10)


# =====================================================================
# 4. Model Provider Factory
# =====================================================================
def get_model(provider: str = "google"):
    """Instantiates the model backend.
    
    To run 100% locally offline, switch provider to 'ollama'.
    """
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0.2,
            timeout=20.0,
            max_retries=1,
            thinking_level="low",
        )
    elif provider == "ollama":
        try:
            return ChatOllama(
                model="llama3.2:3b",
                temperature=0.2,
            )
        except ImportError:
            raise ImportError(
                "Run 'pip install langchain-ollama' to run offline models with Ollama."
            )
    raise ValueError(f"Unsupported model provider: '{provider}'")


# =====================================================================
# 5. Agent 1: Curriculum & Problem Generator
# =====================================================================

THEMES = [
    "farm animals (frogs, ducks, sheep)",
    "space exploration (rockets, stars, planets)",
    "bakery treats (cookies, cupcakes, donuts)",
    "playground games (footballs, marbles, swings)",
    "underwater sea creatures (fish, starfish, turtles)",
    "art & crafting (stickers, beads, stamps)",
]

def generate_next_problem(
    state: StudentSessionState, provider: str = "ollama"
) -> GeneratedProblem:
    """Generates an adaptive math problem tailored to the student's mastery and error history."""
    # 1. Pick a random theme for this specific round
    current_theme = random.choice(THEMES)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are the Pedagogical Content Generator for an adaptive 1st-grade math tutor.\n\n"
                "Pedagogical Rules:\n"
                "1. Novelty Constraint: You are STRICTLY FORBIDDEN from generating any equation in: {completed_equations}.\n"
                "2. Story Theme: You MUST set the story around: {theme}. Do not repeat items from prior turns.\n"
                "3. Dynamic Difficulty Tiers:\n"
                "   - Mastery < 60%: Sums within 10 (e.g., 4 + 3, 6 + 2). Scaffold: 'visual'.\n"
                "   - Mastery 60% - 74%: Sums crossing 10 (e.g., 7 + 4, 8 + 5). Scaffold: 'hint'.\n"
                "   - Mastery >= 75%: Larger equations up to 20 (e.g., 12 + 5, 10 + 2). Scaffold: 'none'.\n"
                "4. Numbers Only: Both operand_a and operand_b MUST be plain integers. No words.\n"
                "5. Story Completeness: Explicitly mention BOTH numbers (operand_a and operand_b) in story_context so the story matches the equation.\n"
                "6. Visual Emoji: Set visual_emoji to a single emoji matching the exact object being counted in the story.\n"
                "NEVER use word labels, variable names, or story nouns (e.g. use 4 and 2, NEVER 'stamp' or 'fuel')."
            ),
        ),
        (
            "user",
            (
                "Student: {name}\n"
                "Mastery Score: {mastery_score}/100\n"
                "Recent Errors: {recent_errors}\n"
                "Completed Equations: {completed_equations}\n"
                "Required Theme: {theme}\n\n"
                "Generate the next unique practice problem."
            ),
        ),
    ])

    llm = get_model(provider=provider)
    if hasattr(llm, "temperature"):
        llm.temperature = 0.7

    structured_llm = llm.with_structured_output(GeneratedProblem)
    chain = prompt | structured_llm

    error_list = (
        [e.value for e in state.recent_errors[-3:]]
        if state.recent_errors
        else ["none"]
    )

    return chain.invoke({
        "name": state.student_name,
        "mastery_score": state.mastery_score,
        "recent_errors": error_list,
        "completed_equations": state.completed_equations or ["none"],
        "theme": current_theme,  # <-- Injected dynamic variable
    })


# =====================================================================
# 6. Agent 2: Diagnostic Evaluator
# =====================================================================
def evaluate_attempt(
    problem: str, student_answer: int, provider: str = "ollama"
) -> DiagnosticOutput:
    """Evaluates student attempt with deterministic equality verification."""
    clean_prob = (
        problem.replace("+=", "+")
        .replace("-=", "-")
        .replace("=?", "")
        .replace("?", "")
        .strip()
    )
    match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", clean_prob)

    if match:
        op_a = int(match.group(1))
        op = match.group(2)
        op_b = int(match.group(3))
        ground_truth = int(
            calculate_arithmetic.invoke({
                "operand_a": op_a,
                "operand_b": op_b,
                "operator": op,
            })
        )
    else:
        # Fallback if equation couldn't be parsed
        print(f"[Warning] Could not extract numeric equation from '{problem}'")
        ground_truth = student_answer

    is_correct = student_answer == ground_truth

    # 2. Fast path: Deterministic success
    if is_correct:
        return DiagnosticOutput(
            is_correct=True,
            diagnosed_error=DiagnosedError.NONE,
            pedagogical_clue="Spot on! You got the right answer!",
            recommended_next_action=RecommendedNextAction.PROCEED,
        )

    # 3. Deterministic Heuristic Extraction for the SLM
    diff = abs(student_answer - ground_truth)
    if diff == 1:
        heuristic_note = (
            "HEURISTIC HINT: The student's answer is off by exactly 1. "
            "This is an 'off_by_one' counting error. Do not classify as reversal."
        )
    elif student_answer == abs(op_a - op_b) and op == "+":
        heuristic_note = (
            "HEURISTIC HINT: The student subtracted instead of adding. "
            "This is an 'operator_confusion' error."
        )
    else:
        heuristic_note = "Analyze the numeric gap to determine the error."

    # 4. LLM Diagnostic Branch
    llm = get_model(provider=provider)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are the diagnostic engine for a 1st-grade math tutor.\n"
                "The student made an incorrect calculation.\n"
                "Classify their misconception into one of the allowed error categories "
                "and give a gentle, encouraging Socratic clue without giving away the answer."
            ),
        ),
        (
            "user",
            (
                "Problem: {problem}\n"
                "Ground Truth: {ground_truth}\n"
                "Student Answer: {student_answer}\n"
                "{heuristic_note}\n\n"
                "Diagnose the misconception."
            ),
        ),
    ])

    structured_llm = llm.with_structured_output(DiagnosticOutput)
    chain = prompt | structured_llm

    result: DiagnosticOutput = chain.invoke({
        "problem": clean_prob,
        "ground_truth": ground_truth,
        "student_answer": student_answer,
        "heuristic_note": heuristic_note,  # <-- Injected mathematical hint
    })

    result.is_correct = False
    return result

# =====================================================================
# 7. Execution Harness (CLI Loop)
# =====================================================================
if __name__ == "__main__":
    # Choose "google" for Gemini 3.8 Flash, or "ollama" for local offline execution
    ACTIVE_PROVIDER = "ollama"

    print("=== Adaptive Math Tutor Initialized ===")
    print(f"Backend Provider: {ACTIVE_PROVIDER.upper()}\n")

    state = StudentSessionState(student_name="Leo")

    for round_num in range(1, 4):
        print(f"\n==================== ROUND {round_num} ====================")
        print(f"Current Mastery: {state.mastery_score}%")

        print("Generating next adaptive problem...")
        problem = generate_next_problem(state, provider=ACTIVE_PROVIDER)

        print(f"\nSkill Focus:  {problem.target_skill}")
        print(f"Scaffolding:  {problem.scaffold_level.value}")
        print(f"Story:        {problem.story_context}")
        print(f"Problem:      {problem.equation}")

        user_input = input("\nStudent Answer (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break

        try:
            student_val = int(user_input)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        print("\nEvaluating attempt...")
        diagnosis = evaluate_attempt(
            problem.equation, student_val, provider=ACTIVE_PROVIDER
        )
        state.record_attempt(problem.equation, diagnosis)

        print("\n--- Diagnostic Feedback ---")
        print(f"Correct:       {diagnosis.is_correct}")
        print(f"Identified:    {diagnosis.diagnosed_error.value}")
        print(f"Tutor Clue:    {diagnosis.pedagogical_clue}")
        print(f"Next Action:   {diagnosis.recommended_next_action.value}")

    print("\nSession complete. Final Mastery Score:", state.mastery_score)