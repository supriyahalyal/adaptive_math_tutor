import { useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000/api";

export default function App() {
  const [studentName, setStudentName] = useState("Leo");
  const [session, setSession] = useState(null);
  const [problem, setProblem] = useState(null);
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchNextProblem = async (sessionId) => {
    setLoading(true);
    setError("");
    setFeedback(null);
    setAnswer("");

    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/next-problem`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to generate next problem.");
      const data = await res.json();
      setProblem(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartSession = async (e) => {
    e.preventDefault();
    if (!studentName.trim()) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_name: studentName.trim(),
          target_skill: "addition_within_20",
        }),
      });
      if (!res.ok) throw new Error("Could not connect to FastAPI backend.");
      const sessionData = await res.json();
      setSession(sessionData);
      await fetchNextProblem(sessionData.session_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleSubmitAnswer = async (e) => {
    e.preventDefault();
    if (answer === "" || !session || !problem) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/sessions/${session.session_id}/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          equation: problem.equation,
          student_answer: parseInt(answer, 10),
        }),
      });
      if (!res.ok) throw new Error("Failed to evaluate answer.");
      const result = await res.json();

      setFeedback(result);
      setSession((prev) => ({
        ...prev,
        mastery_score: result.updated_mastery_score,
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrySameProblem = () => {
    setFeedback(null);
    setAnswer("");
  };

  function VisualScaffolding({ equation, emoji = "🍎" }) {
  const match = equation.match(/(\d+)\s*([+-])\s*(\d+)/);
  if (!match) return null;

  const opA = Math.min(parseInt(match[1], 10), 20);
  const operator = match[2];
  const opB = Math.min(parseInt(match[3], 10), 20);

  if (operator === "+") {
    return (
      <div className="visual-board" aria-label="Visual counting blocks">
        <div className="token-group">
          {Array.from({ length: opA }).map((_, i) => (
            <span key={`a-${i}`} className="token">{emoji}</span>
          ))}
        </div>
        <span className="visual-op">+</span>
        <div className="token-group">
          {Array.from({ length: opB }).map((_, i) => (
            <span key={`b-${i}`} className="token">{emoji}</span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="visual-board" aria-label="Visual subtraction blocks">
      <div className="token-group">
        {Array.from({ length: opA }).map((_, i) => {
          const isRemoved = i >= opA - opB;
          return (
            <span
              key={`sub-${i}`}
              className={`token ${isRemoved ? "token-crossed" : ""}`}
            >
              {emoji}
            </span>
          );
        })}
      </div>
    </div>
  );
}

  return (
    <div className="app-shell">
      <header className="top-bar">
        <h1>1st-Grade Adaptive Math Tutor</h1>
        {session && (
          <div className="mastery-widget">
            <div className="mastery-labels">
              <span>Student: <strong>{session.student_name}</strong></span>
              <span>Mastery: <strong>{session.mastery_score}%</strong></span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${session.mastery_score}%` }}
              />
            </div>
          </div>
        )}
      </header>

      {error && <div className="error-banner">{error}</div>}

      {!session ? (
        <form className="card start-card" onSubmit={handleStartSession}>
          <h2>Start a Practice Session</h2>
          <p>Enter the student&apos;s name to initialize an adaptive learning path.</p>
          <input
            type="text"
            value={studentName}
            onChange={(e) => setStudentName(e.target.value)}
            placeholder="Student name"
            disabled={loading}
          />
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Initializing..." : "Start Learning"}
          </button>
        </form>
      ) : (
        <main className="card problem-card">
          {loading && !problem ? (
            <div className="loader">Generating your first problem...</div>
          ) : (
            problem && (
              <>
                <div className="meta-badges">
                  <span className="badge badge-skill">{problem.target_skill}</span>
                  <span className={`badge badge-scaffold-${problem.scaffold_level}`}>
                    Scaffold: {problem.scaffold_level}
                  </span>
                </div>

                <p className="story-text">{problem.story_context}</p>

                {problem.scaffold_level === "visual" && (
                  <VisualScaffolding
                    equation={problem.equation}
                    emoji={problem.visual_emoji}
                  />
                )}

                {problem.scaffold_level === "hint" && (
                  <div className="strategy-tip">
                    Strategy Tip: Start with the bigger number in your head and count up!
                  </div>
                )}

                <div className="equation-box">{problem.equation} = ?</div>

                {!feedback ? (
                  <form className="answer-form" onSubmit={handleSubmitAnswer}>
                    <input
                      type="number"
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder="?"
                      autoFocus
                      disabled={loading}
                    />
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={loading || answer === ""}
                    >
                      {loading ? "Checking..." : "Check Answer"}
                    </button>
                  </form>
                ) : (
                  <div
                    className={`feedback-box ${
                      feedback.is_correct ? "feedback-correct" : "feedback-wrong"
                    }`}
                  >
                    <div className="feedback-header">
                      <strong>
                        {feedback.is_correct ? "Awesome Job!" : "Let's Look Closer"}
                      </strong>
                      {!feedback.is_correct && (
                        <span className="badge badge-error">
                          Gap: {feedback.diagnosed_error}
                        </span>
                      )}
                    </div>

                    <p className="clue-text">{feedback.pedagogical_clue}</p>

                    <div className="feedback-actions">
                      {!feedback.is_correct && (
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={handleRetrySameProblem}
                        >
                          Try Same Problem Again
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={() => fetchNextProblem(session.session_id)}
                        disabled={loading}
                      >
                        {loading ? "Generating..." : "Next Problem →"}
                      </button>
                    </div>
                  </div>
                )}
              </>
            )
          )}
        </main>
      )}
    </div>
  );
}