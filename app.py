"""Streamlit UI for the Ask Your Data analytics assistant. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="Ask Your Data", page_icon="📊", layout="wide")

from core import (  # noqa: E402
    PROVIDER,
    answer,
    auto_insights,
    check_config,
    df,
    executive_summary,
    log_feedback,
)

if check_config():
    st.error(check_config())
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)


def set_pending(q):
    st.session_state.pending = q


def show_chart(result):
    """Chart the first column (labels) against the last numeric column."""
    if result.shape[1] < 2 or not (1 < len(result) <= 30):
        return
    value_col = result.columns[-1]
    if not pd.api.types.is_numeric_dtype(result[value_col]):
        return
    chart_data = result.set_index(result.columns[0])[[value_col]]
    if pd.api.types.is_numeric_dtype(result.iloc[:, 0]):
        st.line_chart(chart_data)
    else:
        st.bar_chart(chart_data)


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("📊 Campaign overview")
    st.metric("Customers contacted", f"{len(df):,}")
    st.metric("Subscription rate", f"{(df['y'] == 'yes').mean() * 100:.2f}%")
    st.metric("Average balance", f"{df['balance'].mean():,.0f}")
    st.divider()

    st.subheader("Try asking")
    for q in [
        "Which job categories have the highest subscription rate?",
        "How does subscription rate change with number of contacts?",
        "Compare subscription rate by month",
        "Do customers with a housing loan subscribe less?",
    ]:
        st.button(q, on_click=set_pending, args=(q,), key=f"ex_{q}")
    st.divider()

    if st.button("💡 Generate campaign insights"):
        with st.spinner("Analyzing campaign..."):
            st.session_state.auto = auto_insights()
    if st.button("📝 Executive summary of this chat"):
        if st.session_state.history:
            with st.spinner("Summarizing..."):
                st.session_state.summary = executive_summary(st.session_state.history)
        else:
            st.warning("Ask a few questions first.")
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.session_state.pop("summary", None)
        st.session_state.pop("auto", None)
        st.rerun()


# ------------------------------------------------------------ Render a turn
def render_turn(turn, idx, is_last):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        label = "View SQL" + (" (auto-repaired)" if turn.get("repaired") else "")
        with st.expander(label):
            st.code(turn["sql"], language="sql")
        if "error" in turn:
            st.error(turn["error"])
            return

        tab_chart, tab_table = st.tabs(["Chart", "Table"])
        with tab_chart:
            show_chart(turn["result"])
        with tab_table:
            st.dataframe(turn["result"])
            st.download_button(
                "⬇️ Download CSV",
                turn["result"].to_csv(index=False),
                file_name=f"result_{idx}.csv",
                key=f"dl_{idx}",
            )

        st.markdown("**Insight**")
        st.write(turn["insight"])

        c1, c2, _ = st.columns([1, 1, 8])
        if c1.button("👍", key=f"up_{idx}"):
            log_feedback(turn["question"], turn["sql"], "correct")
            st.toast("Thanks, logged as correct")
        if c2.button("👎", key=f"down_{idx}"):
            log_feedback(turn["question"], turn["sql"], "incorrect")
            st.toast("Thanks, logged as incorrect")

        if is_last and turn.get("followups"):
            st.caption("Suggested follow-ups")
            for j, f in enumerate(turn["followups"]):
                st.button(f, on_click=set_pending, args=(f,), key=f"fu_{idx}_{j}")


# -------------------------------------------------------------------- Main
st.title("Ask Your Data: Bank Marketing Campaign")
st.caption(f"Ask in plain English. The app writes the SQL, runs it, charts the result and explains it. (LLM provider: {PROVIDER})")

if st.session_state.get("auto"):
    st.success(st.session_state.auto)
if st.session_state.get("summary"):
    st.info(st.session_state.summary)

question = st.chat_input("Ask a question about the campaign data...")
if st.session_state.pending:
    question, st.session_state.pending = st.session_state.pending, None

history = st.session_state.history
for i, t in enumerate(history):
    render_turn(t, i, is_last=(i == len(history) - 1 and not question))

if question:
    with st.spinner("Analyzing..."):
        try:
            new_turn = answer(question, history)
        except Exception as e:
            new_turn = {"question": question, "sql": "-- no query generated", "error": f"LLM call failed: {e}"}
    history.append(new_turn)
    render_turn(new_turn, len(history) - 1, is_last=True)
