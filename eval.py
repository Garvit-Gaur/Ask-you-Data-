"""Accuracy test: run 10 questions through the pipeline and compare with answers computed in pandas.

Run:  python eval.py
Output: a results table in the terminal + eval_results.csv
"""
import pandas as pd

from core import df, generate_sql, is_safe, run_query


def rate(d):
    return round((d["y"] == "yes").mean() * 100, 2)


def top_rate(col):
    return round(df.groupby(col)["y"].apply(lambda s: (s == "yes").mean() * 100).max(), 2)


# (question, expected number, tolerance). Expected values are computed from the raw data.
CASES = [
    ("What is the overall subscription rate?", rate(df), 0.05),
    ("How many customers are in the dataset?", len(df), 0.5),
    ("What is the average balance of all customers?", round(df["balance"].mean(), 2), 0.5),
    ("How many customers have a housing loan?", int((df["housing"] == "yes").sum()), 0.5),
    ("What is the subscription rate for customers with a personal loan?", rate(df[df["loan"] == "yes"]), 0.05),
    ("What is the average age of customers who subscribed?", round(df[df["y"] == "yes"]["age"].mean(), 2), 0.05),
    ("How many customers were contacted in May?", int((df["month"] == "may").sum()), 0.5),
    ("What is the average call duration in seconds for customers who subscribed?", round(df[df["y"] == "yes"]["duration"].mean(), 2), 0.5),
    ("Which job category has the highest subscription rate, and what is that rate?", top_rate("job"), 0.05),
    ("Which month has the highest subscription rate, and what is that rate?", top_rate("month"), 0.05),
]


def contains_value(result, expected, tol):
    """True if any number in the result table is within tol of the expected value."""
    values = pd.to_numeric(pd.Series(result.to_numpy().ravel()), errors="coerce").dropna()
    return bool(((values - expected).abs() <= tol).any())


rows = []
for question, expected, tol in CASES:
    sql, error, ok = generate_sql(question), "", False
    if not is_safe(sql):
        error = "blocked by safety check"
    else:
        try:
            ok = contains_value(run_query(sql), expected, tol)
        except Exception as e:
            error = str(e)[:80]
    rows.append({"question": question, "expected": expected, "correct": ok, "error": error, "sql": sql})
    print(f"{'PASS' if ok else 'FAIL'}  {question}  (expected {expected})  {error}")

report = pd.DataFrame(rows)
report.to_csv("eval_results.csv", index=False)
print(f"\nAccuracy: {int(report['correct'].sum())}/{len(report)}")
print("Open eval_results.csv and read the SQL of every FAIL to see why it failed.")
