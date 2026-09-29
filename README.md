# Ask Your Data: LLM-Powered Analytics Assistant

A conversational analytics app that lets business users ask questions about a bank's marketing campaign in plain English. The app writes the SQL, validates it, runs it, charts the result, and explains the finding in plain language.

![screenshot](screenshot.png)  <!-- replace with your screenshot -->

## Why I built it
Business stakeholders often wait on analysts for simple campaign questions. This tool lets them self-serve while keeping the analysis auditable: every answer shows the SQL that produced it.

## How it works
1. **Question** typed by the user (multi-turn: follow-ups like "now break that down by month" work).
2. **SQL generation:** the LLM receives the table schema, sample rows and data notes, and returns one SELECT query.
3. **Validation:** only read-only SELECT/WITH queries pass; the database connection is also set to `query_only`.
4. **Execution + auto-repair:** if the query fails, the error is sent back to the LLM for one corrected retry.
5. **Presentation:** table, auto-chart, CSV download, and a short insight (finding, caveat, next step).
6. **Extras:** suggested follow-up questions, campaign-level auto-insights, executive summary of the session, 👍/👎 feedback logging.

**Stack:** Python, pandas, SQLite, Streamlit, Claude API (Anthropic).

## Dataset
[Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-full.csv`, about 45K customers contacted in a phone campaign. Not included in this repo; download it and place it in the project folder.

## Run it locally
```bash
git clone https://github.com/YOUR_USERNAME/ask-your-data.git
cd ask-your-data
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then paste your ANTHROPIC_API_KEY into .env
# put bank-full.csv in this folder
streamlit run app.py
```

## Evaluation
`python eval.py` runs 10 questions and compares the generated SQL's output to answers I computed independently in pandas.

| Run | Correct |
|---|---|
| First pass | X / 10 |
| After prompt improvements | Y / 10 |

**Failures I found and fixed:** (fill in, e.g. "ambiguous 'rate' definition → added the rate formula to the prompt", "reserved word `default` needed quoting → added a data note")

## Limitations
- The LLM can still generate incorrect SQL; that is why the SQL is always shown and feedback is logged.
- Single table only; no joins.
- The eval checks whether the expected number appears in the result, so I also read the SQL of each case manually.

## Ideas for next steps
Multiple tables, query caching, logging to a database, an evaluation dashboard page.
