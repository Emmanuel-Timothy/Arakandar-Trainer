"""Purpose-built Streamlit analyst workspace for the Arakandar agent."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.agent.orchestrator import build_default_agent
from app.model_registry import ModelRegistry
from app.paper_trading import PaperBroker


st.set_page_config(page_title="Arakandar Analyst", page_icon="A", layout="wide")
st.title("Arakandar Analyst")
st.caption("Custom agent workflow for market evidence, signals, and paper-only research.")

agent = build_default_agent()
registry = ModelRegistry(agent.store, agent.model_path)
paper_broker = PaperBroker(agent.store)
raw_files = sorted(agent.data_dir.glob("*_ohlcv.csv"))
tickers = [path.stem.removesuffix("_ohlcv").replace("_", ".") for path in raw_files]

with st.sidebar:
    st.header("Task")
    ticker = st.selectbox("Ticker", tickers or ["BBCA.JK"])
    headlines_text = st.text_area(
        "Headlines",
        value="BBCA reports record profit and dividend hike",
        help="One headline per line. These are scored locally by the news tokenizer.",
    )
    run_task = st.button("Explain latest signal", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Recent runs")
    for task in agent.store.recent_tasks(limit=5):
        st.caption(f"{task['status'].upper()} · {task['task'][:42]}")

if run_task:
    with st.status("Executing analyst workflow", expanded=True) as status:
        st.write("Routing through market snapshot, news sentiment, prediction, and report tools.")
        try:
            result = agent.run(
                "Explain the latest signal",
                ticker,
                [line.strip() for line in headlines_text.splitlines() if line.strip()],
            )
            status.update(label="Workflow complete", state="complete")
        except Exception as exc:
            status.update(label="Workflow failed", state="error")
            st.exception(exc)
            result = None

    if result:
        report = result["report"]
        st.subheader(report["headline"])
        st.info(report["summary"])
        left, right = st.columns(2)
        with left:
            st.metric("Signal", result["report"]["headline"].split(": ")[-1])
            st.write("**Evidence**")
            st.json(report["evidence"])
        with right:
            st.write("**Execution plan**")
            st.write(" → ".join(result["plan"]))
            st.write("**Source route**")
            st.json(result["route"])
        st.warning(report["disclaimer"])

        st.subheader("Tool trace")
        events = agent.store.task_events(result["task_id"])
        st.dataframe(
            [
                {
                    "tool": event["tool_name"],
                    "status": event["status"],
                    "started_at": event["started_at"],
                    "error": event["error"] or "",
                }
                for event in events
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Raw workflow result"):
            st.code(json.dumps(result, indent=2, default=str), language="json")
else:
    st.subheader("Ready")
    st.write("Select a ticker and run an evidence-backed signal explanation.")
    st.write("The workflow is paper-only and records every tool call in local SQLite state.")

st.divider()
st.subheader("Paper portfolio")
positions = paper_broker.store.paper_positions()
if positions:
    st.dataframe(positions, use_container_width=True, hide_index=True)
else:
    st.caption("No paper positions yet.")
with st.form("paper_order"):
    order_side = st.selectbox("Side", ["BUY", "SELL"])
    order_quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
    order_price = st.number_input("Fill price", min_value=0.01, value=100.0, step=0.01)
    submit_order = st.form_submit_button("Submit paper order")
    if submit_order:
        try:
            paper_broker.submit_market_order(ticker, order_side, order_quantity, order_price)
            st.success("Paper order filled.")
        except ValueError as exc:
            st.error(str(exc))

st.divider()
st.subheader("Model registry")
versions = registry.store.model_versions()
if versions:
    st.dataframe(
        [
            {
                "version": version["version_id"],
                "status": version["status"],
                "created_at": version["created_at"],
                "metrics": version["metrics_json"],
            }
            for version in versions
        ],
        use_container_width=True,
        hide_index=True,
    )
    candidates = [version for version in versions if version["status"] == "candidate"]
    if candidates:
        selected = st.selectbox("Candidate", [candidate["version_id"] for candidate in candidates])
        if st.button("Approve candidate"):
            registry.approve(selected)
            st.success("Candidate approved and copied to champion.")
else:
    st.caption("No candidate model versions registered yet.")
