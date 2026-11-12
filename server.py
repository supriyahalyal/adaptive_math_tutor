"""FastAPI HTTP Wrapper for tutor_agent.py."""

import os
import uuid
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import core domain models, state machine, and agents directly from tutor_agent.py
from tutor_agent import (
    DiagnosedError,
    RecommendedNextAction,
    ScaffoldLevel,
    StudentSessionState,
    evaluate_attempt,
    generate_next_problem,
)

# =====================================================================
# 1. Configuration & In-Memory Session Store
# =====================================================================
ACTIVE_PROVIDER = os.getenv("TUTOR_PROVIDER", "ollama")
SESSION_STORE: dict[str, StudentSessionState] = {}

app = FastAPI(
    title="Adaptive 1st-Grade Math Tutor API",
    version="1.0.0",
    description="REST API wrapper around the local LangChain tutoring engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 2. HTTP Request & Response Contracts
# =====================================================================
class CreateSessionRequest(BaseModel):
    student_name: str = Field(..., example="Leo")
    target_skill: str = Field("addition_within_20", example="addition_within_20")


class SessionResponse(BaseModel):
    session_id: str
    student_name: str
    target_skill: str
    mastery_score: int
    completed_equations: list[str]


class NextProblemResponse(BaseModel):
    equation: str
    story_context: str
    visual_emoji: str
    scaffold_level: ScaffoldLevel
    target_skill: str
    current_mastery: int


class SubmitAttemptRequest(BaseModel):
    equation: str = Field(..., example="5 + 2")
    student_answer: int = Field(..., example=7)


class SubmitAttemptResponse(BaseModel):
    is_correct: bool
    diagnosed_error: DiagnosedError
    pedagogical_clue: str
    recommended_next_action: RecommendedNextAction
    updated_mastery_score: int


# =====================================================================
# 3. API Routes
# =====================================================================
@app.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize a new student session",
)
def create_session(payload: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    state = StudentSessionState(
        student_name=payload.student_name,
        target_skill=payload.target_skill,
    )
    SESSION_STORE[session_id] = state

    return SessionResponse(
        session_id=session_id,
        student_name=state.student_name,
        target_skill=state.target_skill,
        mastery_score=state.mastery_score,
        completed_equations=state.completed_equations,
    )


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get current student mastery and history",
)
def get_session(session_id: str):
    state = SESSION_STORE.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    return SessionResponse(
        session_id=session_id,
        student_name=state.student_name,
        target_skill=state.target_skill,
        mastery_score=state.mastery_score,
        completed_equations=state.completed_equations,
    )


@app.post(
    "/api/sessions/{session_id}/next-problem",
    response_model=NextProblemResponse,
    summary="Generate the next adaptive math problem",
)
def api_next_problem(session_id: str):
    state = SESSION_STORE.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    problem = generate_next_problem(state, provider=ACTIVE_PROVIDER)

    return NextProblemResponse(
        equation=problem.equation,
        story_context=problem.story_context,
        visual_emoji=problem.visual_emoji,
        scaffold_level=problem.scaffold_level,
        target_skill=problem.target_skill,
        current_mastery=state.mastery_score,
    )


@app.post(
    "/api/sessions/{session_id}/submit",
    response_model=SubmitAttemptResponse,
    summary="Evaluate student answer and update session state",
)
def api_submit_attempt(session_id: str, payload: SubmitAttemptRequest):
    state = SESSION_STORE.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    diagnosis = evaluate_attempt(
        payload.equation, payload.student_answer, provider=ACTIVE_PROVIDER
    )
    state.record_attempt(payload.equation, diagnosis)

    return SubmitAttemptResponse(
        is_correct=diagnosis.is_correct,
        diagnosed_error=diagnosis.diagnosed_error,
        pedagogical_clue=diagnosis.pedagogical_clue,
        recommended_next_action=diagnosis.recommended_next_action,
        updated_mastery_score=state.mastery_score,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)