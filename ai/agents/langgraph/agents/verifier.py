"""VerifierAgent — LLM structured judgment + LLM-as-judge for text answers + interrupt P2."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langgraph.types import interrupt

log = logging.getLogger(__name__)


def _verify_learner_answer(
    user_message: str, rag_docs: list, question_context: str = "", model: str = None
) -> Dict[str, Any]:
    """Verify the learner's answer using Python arithmetic, RAG, then web search.

    question_context: last assistant message — helps the RAG heuristic know what was asked.
    Returns a dict: correct (bool), correct_answer (str), explanation (str), method (str)
    Returns {} if no verifiable answer is detected in the message.
    """
    import re

    # Normalise operators
    normalised = (
        user_message.replace("×", "*")
        .replace("÷", "/")
        .replace("x", "*")
        .replace("X", "*")
    )

    # ── Method 1 : Python arithmetic ──────────────────────────────────────────
    # Only for near-bare expressions. On prose with several sentences, the regex
    # can grab an unrelated "= number" fragment (e.g. mid-explanation) and eval()
    # it out of context, producing a confidently wrong verdict instead of falling
    # through to the LLM judge (Method 2) which handles natural language correctly.
    prose_words = re.findall(r"[A-Za-z]{2,}", user_message)
    match = (
        re.search(r"([\d\s\+\-\*\/\(\)\.]+)\s*=\s*([\d\.]+)", normalised)
        if len(prose_words) <= 3
        else None
    )
    if match:
        expression = match.group(1).strip()
        learner_str = match.group(2).strip()
        try:
            correct_val = eval(expression, {"__builtins__": {}})
            correct_rounded = (
                int(correct_val)
                if correct_val == int(correct_val)
                else round(correct_val, 4)
            )
            is_correct = abs(float(learner_str) - float(correct_rounded)) < 0.01
            return {
                "correct": is_correct,
                "learner_answer": learner_str,
                "correct_answer": str(correct_rounded),
                "explanation": (
                    f"{expression} = {correct_rounded}"
                    if is_correct
                    else f"{expression} = {correct_rounded}, pas {learner_str}"
                ),
                "method": "python",
            }
        except Exception:
            pass

    # ── Method 2 : LLM-as-judge for textual answers ──────────────────────────
    if question_context:
        try:
            from ai.llm.service import call_llm

            corpus = (
                " ".join(d.get("content", "") for d in rag_docs)[:600]
                if rag_docs
                else ""
            )
            rag_section = (
                f"\nContenu du cours (référence) :\n{corpus}" if corpus else ""
            )

            prompt = (
                f"Tu es un correcteur pédagogique. Évalue la réponse de l'apprenant.\n\n"
                f"Question posée : {question_context[:300]}\n"
                f"Réponse de l'apprenant : {user_message.strip()[:300]}"
                f"{rag_section}\n\n"
                f"Réponds UNIQUEMENT en JSON :\n"
                f'{{"verdict": "correct|partial|wrong", '
                f'"explanation": "explication courte", '
                f'"correct_answer": "réponse attendue ou vide si correct"}}'
            )

            llm_text = call_llm(prompt, model=model, max_tokens=200)
            if llm_text:
                start = llm_text.find("{")
                end = llm_text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(llm_text[start:end])
                    verdict = data.get("verdict", "")
                    correct_map = {"correct": True, "partial": None, "wrong": False}
                    if verdict in correct_map:
                        return {
                            "correct": correct_map[verdict],
                            "learner_answer": user_message.strip(),
                            "correct_answer": data.get("correct_answer", ""),
                            "explanation": data.get("explanation", ""),
                            "method": "llm_judge",
                        }
        except Exception as exc:
            log.debug("LLM-as-judge failed: %s", exc)

    return {}


def verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    exercises = list(state.get("exercises") or [])
    strategy = state.get("strategy", "")
    rag_docs = list(state.get("rag_docs") or [])
    human_feedback = state.get("human_feedback", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})
    n_retries = dict(state.get("n_retries") or {})
    verification_feedback = list(state.get("verification_feedback") or [])
    answer_history = list(state.get("answer_history") or [])

    # Extract last assistant message to give verifier the question context
    messages = list(state.get("messages") or [])
    last_question = ""
    for m in reversed(messages[:-1]):  # skip last entry (learner's current answer)
        if m.get("role") == "assistant":
            last_question = m.get("content", "")[:300]
            break

    from ai.agents.helpers import is_text_supported
    from ai.llm.service import call_llm

    try:
        from config import settings

        threshold = settings.CONTEXT_RETRIEVAL_CONFIG.get("rag", {}).get(
            "verification_threshold", 0.65
        )
    except Exception:
        threshold = 0.65

    corpus = " ".join(d.get("content", "") for d in rag_docs)
    verification: Dict[str, Any]

    if not rag_docs:
        verification = {
            "verdict": "no_sources",
            "score": 0.0,
            "specific_feedback": [],
            "unsupported_items": [],
        }
        agent_reasoning["verifier"] = "[no_sources] No RAG docs available"
    else:
        exercises_summary = json.dumps(exercises[:3], ensure_ascii=False)[:500]
        prompt = (
            f"Vérifie que les exercices et la stratégie sont supportés par les sources RAG.\n\n"
            f"Stratégie : {strategy[:200]}\n"
            f"Exercices : {exercises_summary}\n"
            f"Sources RAG : {corpus[:500]}\n\n"
            f"Réponds en JSON :\n"
            f'{{"verdict": "supported|needs_review|no_sources", "score": 0.0, '
            f'"specific_feedback": ["..."], "unsupported_items": ["..."]}}'
        )
        llm_text = call_llm(prompt, model=state.get("model"), max_tokens=300)

        if llm_text:
            try:
                start = llm_text.index("{")
                end = llm_text.rindex("}") + 1
                verification = json.loads(llm_text[start:end])
                agent_reasoning["verifier"] = (
                    f"[LLM] verdict={verification.get('verdict')} "
                    f"score={verification.get('score', 0):.2f}"
                )
            except Exception:
                verification = _text_overlap_verdict(
                    strategy, exercises, corpus, threshold
                )
                agent_reasoning["verifier"] = "[fallback] text-overlap"
        else:
            verification = _text_overlap_verdict(strategy, exercises, corpus, threshold)
            agent_reasoning["verifier"] = "[fallback] text-overlap"

    # Propagate specific feedback to planner on next retry
    if verification.get("specific_feedback"):
        verification_feedback = list(verification_feedback) + list(
            verification["specific_feedback"]
        )

    # ── Learner answer verification ───────────────────────────────────────────
    user_message = state.get("user_message", "")
    learner_answer_verdict = _verify_learner_answer(
        user_message, rag_docs, last_question, model=state.get("model")
    )
    if learner_answer_verdict:
        method = learner_answer_verdict.get("method", "?")
        correct = learner_answer_verdict.get("correct")
        agent_reasoning["verifier_answer"] = (
            f"[{method}] learner={'correct' if correct else 'wrong' if correct is False else 'uncertain'} "
            f"— {learner_answer_verdict.get('explanation', '')[:80]}"
        )
        # Accumulate verdict into session answer history
        answer_history = answer_history + [
            {
                **learner_answer_verdict,
                "user_message": user_message[:100],
            }
        ]

    agent_trace.append(
        f"verifier → verdict={verification.get('verdict')} "
        f"score={verification.get('score', 0.0):.2f} "
        f"| answer_check={'yes' if learner_answer_verdict else 'n/a'}"
    )

    # Human-in-the-Loop P2 — ask learner when items cannot be verified
    if verification.get("verdict") == "needs_review" and not human_feedback:
        unsupported = verification.get("unsupported_items", [])
        human_feedback = interrupt(
            {
                "checkpoint": "P2",
                "message": (
                    f"Éléments non vérifiés dans les sources RAG : {unsupported}. "
                    f"Continuer quand même ? (oui / non)"
                ),
                "verification": verification,
            }
        )
        if not isinstance(human_feedback, str):
            human_feedback = ""

    return {
        "verification": verification,
        "verification_feedback": verification_feedback,
        "learner_answer_verdict": learner_answer_verdict,
        "answer_history": answer_history,
        "human_feedback": human_feedback,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
        "n_retries": n_retries,
    }


def _text_overlap_verdict(
    strategy: str,
    exercises: list,
    corpus: str,
    threshold: float,
) -> Dict[str, Any]:
    from ai.agents.helpers import is_text_supported

    text = strategy + " ".join(e.get("question", "") for e in exercises)
    supported = is_text_supported(text, corpus, threshold)
    return {
        "verdict": "supported" if supported else "needs_review",
        "score": 0.8 if supported else 0.4,
        "specific_feedback": [],
        "unsupported_items": [],
    }
