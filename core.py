"""Core logic: data loading + LLM pipeline.

No Streamlit code lives here, so the same functions are reused by the
web app (app.py) and the accuracy test script (eval.py).
"""
import json
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Which LLM to use: "gemini" (free tier), "ollama" (free, runs on your PC) or "anthropic" (paid)
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",       # free, no card; also try "qwen-3-32b"
    "openrouter": "deepseek/deepseek-chat",  # free, no card; DeepSeek via OpenRouter
    "ollama": "qwen2.5-coder:7b",            # free, fully local
    "anthropic": "claude-sonnet-5",          # paid
}
MODEL = os.getenv("MODEL_NAME", DEFAULT_MODELS.get(PROVIDER, ""))
DATA_PATH = os.getenv("DATA_PATH", "bank-full.csv")


def check_config():
    """Return an error message if the chosen provider is not set up, else None."""
    if PROVIDER not in DEFAULT_MODELS:
        return f"LLM_PROVIDER must be one of {list(DEFAULT_MODELS)}, got '{PROVIDER}'."
    if PROVIDER == "gemini" and not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY missing in .env (free key: aistudio.google.com)."
    if PROVIDER == "groq" and not os.getenv("GROQ_API_KEY"):
        return "GROQ_API_KEY missing in .env (free key: console.groq.com/keys)."
    if PROVIDER == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY missing in .env (free key: openrouter.ai/keys)."
    if PROVIDER == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY missing in .env."
    return None


# ---------------------------------------------------------------------------
# 1. Load the CSV into an in-memory SQL database
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")
conn = sqlite3.connect(":memory:", check_same_thread=False)
df.to_sql("bank", conn, index=False)
conn.execute("PRAGMA query_only = ON")  # database-level guard: no writes possible

# The schema is what lets the LLM write correct SQL instead of guessing columns
SCHEMA = (
    "Table: bank\nColumns:\n"
    + "\n".join(f"- {c} ({t})" for c, t in df.dtypes.astype(str).items())
    + "\n\nSample rows:\n"
    + df.head(3).to_string()
)

DATA_NOTES = """
- 'y' is 'yes'/'no': whether the customer subscribed to a term deposit.
- 'campaign' = number of contacts during this campaign; 'duration' = last call length in seconds.
- 'poutcome' = outcome of the previous campaign; 'previous' = contacts before this campaign.
- 'month' values are lowercase 3-letter names (jan, feb, ..., dec).
- 'default' is a reserved word in SQL: always write it as "default" in double quotes.
"""


# ---------------------------------------------------------------------------
# 2. LLM helper
# ---------------------------------------------------------------------------
_client = None  # created on first use, so unused providers need no key or package


def call_llm(system, user, max_tokens=800):
    """One function talks to the LLM. Swap providers via LLM_PROVIDER in .env."""
    global _client
    if PROVIDER == "gemini":
        from google import genai

        if _client is None:
            _client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"timeout": 30_000},  # ms; fail fast instead of hanging
            )
        interaction = _client.interactions.create(
            model=MODEL,
            input=user,
            system_instruction=system,
            generation_config={
                "max_output_tokens": max_tokens,
                "thinking_level": "low",  # keep latency down; "medium" is the (slow) default
            },
        )
        return (interaction.output_text or "").strip()

    if PROVIDER in ("groq", "openrouter"):
        from openai import OpenAI

        if _client is None:
            base_url, key = {
                "groq": ("https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY")),
                "openrouter": ("https://openrouter.ai/api/v1", os.getenv("OPENROUTER_API_KEY")),
            }[PROVIDER]
            _client = OpenAI(base_url=base_url, api_key=key)
        resp = _client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    if PROVIDER == "ollama":
        import urllib.request

        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()

    
    if _client is None:
        _client = Anthropic()
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def clean_sql(text):
    text = re.sub(r"```(?:sql)?", "", text).strip()
    return text.rstrip(";").strip()


# ---------------------------------------------------------------------------
# 3. Question -> SQL
# ---------------------------------------------------------------------------
def generate_sql(question):
    system = f"""You are a data analyst writing SQLite queries.
{SCHEMA}
{DATA_NOTES}
Rules:
- Return ONLY one SELECT query. No explanation, no markdown fences.
- Use only the table and columns above.
- For rates, use ROUND(100.0 * SUM(CASE WHEN y='yes' THEN 1 ELSE 0 END) / COUNT(*), 2).
- If the question refers to earlier questions ("that", "now break it down"), use the previous questions for context.
- Limit results to 50 rows unless asked otherwise."""
    return clean_sql(call_llm(system, question))


def repair_sql(question, bad_sql, error):
    system = f"""You are a data analyst fixing a broken SQLite query.
{SCHEMA}
{DATA_NOTES}
Return ONLY the corrected single SELECT query. No explanation, no markdown."""
    user = f"Question: {question}\nFailed SQL: {bad_sql}\nError: {error}"
    return clean_sql(call_llm(system, user))


# ---------------------------------------------------------------------------
# 4. Safety: never run LLM-generated SQL blindly
# ---------------------------------------------------------------------------
BANNED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b",
    re.IGNORECASE,
)


def is_safe(sql):
    s = sql.strip().lower()
    return s.startswith(("select", "with")) and ";" not in s and not BANNED.search(s)


def run_query(sql):
    return pd.read_sql_query(sql, conn)


# ---------------------------------------------------------------------------
# 5. Result -> plain-English insight, follow-ups, summaries
# ---------------------------------------------------------------------------
def explain(question, sql, result_df):
    system = """You are a senior analyst talking to a non-technical business stakeholder.
Explain the result in 3-4 sentences: the key finding, one caveat, and one suggested next step.
Only use numbers that appear in the result. Do not invent figures."""
    user = f"Question: {question}\nSQL: {sql}\nResult:\n{result_df.head(20).to_string()}"
    return call_llm(system, user)


def suggest_followups(question, result_df):
    system = """You are a marketing analytics lead. Suggest 3 short follow-up questions
a business user could ask next about this data. Return ONLY a JSON list of 3 strings."""
    raw = call_llm(
        system,
        f"Question: {question}\nResult:\n{result_df.head(10).to_string()}",
        max_tokens=300,
    )
    try:
        return json.loads(re.sub(r"```(?:json)?", "", raw).strip())[:3]
    except Exception:
        return []


def executive_summary(history):
    system = """You are a senior analyst writing a brief for leadership.
From the Q&A below, write: 1) a 2-sentence overview, 2) top 3 findings, 3) 2 recommended actions.
Use only facts from the insights provided."""
    text = "\n\n".join(
        f"Q: {t['question']}\nInsight: {t['insight']}" for t in history if "insight" in t
    )
    return call_llm(system, text, max_tokens=900)


def auto_insights():
    queries = {
        "By job": "SELECT job, COUNT(*) AS n, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS rate FROM bank GROUP BY job ORDER BY rate DESC",
        "By month": "SELECT month, COUNT(*) AS n, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS rate FROM bank GROUP BY month ORDER BY rate DESC",
        "By previous outcome": "SELECT poutcome, COUNT(*) AS n, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS rate FROM bank GROUP BY poutcome ORDER BY rate DESC",
    }
    text = "\n\n".join(f"{k}:\n{run_query(q).to_string()}" for k, q in queries.items())
    system = (
        "You are a marketing analytics lead. Give the top 3 findings and 2 recommendations "
        "for improving campaign performance. Use only the numbers provided."
    )
    return call_llm(system, text, max_tokens=1000)


# ---------------------------------------------------------------------------
# 6. The full pipeline for one question
# ---------------------------------------------------------------------------
def answer(question, history):
    """question -> SQL -> validate -> run (with one auto-repair) -> insight + follow-ups."""
    context = "\n".join(f"Q: {t['question']}\nSQL: {t['sql']}" for t in history[-3:])
    full_q = (
        f"Previous questions:\n{context}\n\nNew question: {question}" if context else question
    )

    sql = generate_sql(full_q)
    turn = {"question": question, "sql": sql}

    if not is_safe(sql):
        turn["error"] = "Blocked: only read-only SELECT queries are allowed."
        return turn

    try:
        result = run_query(sql)
    except Exception as e:  # self-repair: send the error back to the LLM once
        sql = repair_sql(question, sql, str(e))
        turn["sql"], turn["repaired"] = sql, True
        if not is_safe(sql):
            turn["error"] = "Blocked after repair attempt."
            return turn
        try:
            result = run_query(sql)
        except Exception as e2:
            turn["error"] = f"Query failed even after repair: {e2}"
            return turn

    turn["result"] = result
    turn["insight"] = explain(question, sql, result)
    turn["followups"] = suggest_followups(question, result)
    return turn


# ---------------------------------------------------------------------------
# 7. Feedback logging (thumbs up / down from the UI)
# ---------------------------------------------------------------------------
def log_feedback(question, sql, verdict, path="feedback_log.csv"):
    row = pd.DataFrame(
        [{"time": datetime.now().isoformat(), "question": question, "sql": sql, "verdict": verdict}]
    )
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
