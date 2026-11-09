"""Prototype Streamlit: MCTS, MCTS + EDD, dan MCTS + LDD untuk Job Shop Scheduling.

Jalankan:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from jssp_core import POLICY_LABELS, JSSProblem, run_single, summarize

st.set_page_config(page_title="MCTS with Heuristic for JSSP", layout="wide")
st.title("Monte Carlo Tree Search for Job Shop Scheduling (with Heuristic : Dispatching Rule)")

col_mcts, col_edd, col_ldd = st.columns(3)
with col_mcts:
    st.markdown(
        "**Monte Carlo Tree Search (MCTS)**\n\n"
        "MCTS builds the search tree incrementally through four steps: *selection* "
        "(choosing the node with the best UCB value), *expansion* (adding child nodes representing "
        "schedulable operations), *simulation* (completing the schedule until all operations are scheduled), "
        "and *backpropagation* (updating node values with reward = −Cmax). In the baseline, "
        "the simulation selects jobs at random."
    )
with col_edd:
    st.markdown(
        "**MCTS + EDD (Earliest Due Date)**\n\n"
       "The simulation step is guided by a heuristic. Candidate operations are restricted to the "
        "Giffler-Thompson *conflict set*, and the job with the earliest *due date* is then selected. Due dates are computed using "
        "the TWK method: d = ⌈1.5 × total job processing time⌉. With probability ε = 0.1, a candidate is selected "
        "at random to reduce deterministik."
    )
with col_ldd:
    st.markdown(
        "**MCTS + LDD (Latest Due Date)**\n\n"
        "As with EDD, candidate operations are drawn from the Giffler-Thompson *conflict set*, with "
        "due dates computed using the TWK method and ε = 0.1. The difference is that LDD prioritizes the job with the latest "
        "*due date*, the job with the largest total processing time."
    )
st.divider()

# ---------------------------------------------------------------------------
# 1. Input dataset
# ---------------------------------------------------------------------------
st.sidebar.header("1. Dataset")
fmt = "auto"  # format Taillard / OR-Library dideteksi otomatis
text, name = None, None
up = st.sidebar.file_uploader(
    "Upload file instance (.txt)", type=["txt"],
)
if up is not None:
    text, name = up.getvalue().decode("utf-8", errors="ignore"), up.name

# ---------------------------------------------------------------------------
# 2. Parameter
# ---------------------------------------------------------------------------
st.sidebar.header("2. Parameter")
policies = st.sidebar.multiselect(
    "Method",
    options=list(POLICY_LABELS),
    default=["random", "edd", "ldd"],
    format_func=POLICY_LABELS.get,
)
iters_text = st.sidebar.text_input("Number of iterations (run_num), comma-separated", "100")
num_runs = st.sidebar.number_input("Number of runs per configuration (NUM_RUNS)", min_value=5, max_value=20, value=5, step=1)

# Parameter tetap (sesuai notebook eksperimen)
BASE_SEED = 0
EXPLORATION_WEIGHT = 2
EPSILON = 0.1
DUE_DATE_FACTOR = 1.5

# ---------------------------------------------------------------------------
# Parse instance
# ---------------------------------------------------------------------------
if not text or not text.strip():
    st.info("Upload a JSSP instance file in the sidebar to get started.")
    st.stop()

try:
    problem = JSSProblem(text, fmt=fmt, due_date_factor=DUE_DATE_FACTOR, name=name)
except ValueError as e:
    st.error(f"Fail to read instance: {e}")
    st.stop()

st.subheader(f"Instance: {name}")
st.table(pd.DataFrame(
    {"Jumlah": [problem.num_jobs, problem.num_machines, problem.num_jobs * problem.num_steps]},
    index=["Job", "Machine", "Operation"],
))
for w in problem.warnings[:5]:
    st.warning(w)

with st.expander("Instance Data"):
    rows = []
    for j, ops in enumerate(problem.jobs):
        rows.append({
            "Job": j,
            "machine sequence": " → ".join(str(m) for m, _ in ops),
            "Duration": " ".join(str(d) for _, d in ops),
            "Total duration": sum(d for _, d in ops),
            "Due date": problem.due_dates[j],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Jalankan eksperimen
# ---------------------------------------------------------------------------
try:
    run_nums = sorted({int(x) for x in iters_text.replace(" ", "").split(",") if x})
    if not run_nums or min(run_nums) <100 or max(run_nums) > 3000:
        raise ValueError
except ValueError:
    st.error("The number of iterations must be a positive integer between 100 - 3000")
    st.stop()

if not policies:
    st.warning("Choose method.")
    st.stop()

total_jobs = len(policies) * len(run_nums) * num_runs
st.write(f"Total runs to be executed: **{total_jobs}** "
         f"({len(policies)} method × {len(run_nums)} run_num × {num_runs} run).")

if st.button("Run", type="primary"):
    results = []
    bar = st.progress(0.0)
    status = st.empty()
    done = 0
    for policy in policies:
        for run_num in run_nums:
            for i in range(num_runs):
                seed = BASE_SEED + i
                status.write(f"{POLICY_LABELS[policy]} | run_num={run_num} | seed={seed} ...")

                def cb(frac, done=done):
                    bar.progress(min(1.0, (done + frac) / total_jobs))

                results.append(run_single(problem, run_num, seed, EXPLORATION_WEIGHT, EPSILON, policy, cb))
                done += 1
                bar.progress(done / total_jobs)
    status.success("Done.")
    st.session_state["results"] = results
    st.session_state["results_key"] = (name, text, fmt)

# ---------------------------------------------------------------------------
# Tampilkan hasil
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if not results:
    st.stop()
if st.session_state.get("results_key") != (name, text, fmt):
    st.warning("The instance has changed since the last run. The results below are from the previous configuration.")

df = pd.DataFrame([{
    "Method": POLICY_LABELS[r["policy"]],
    "run_num": r["run_num"],
    "Seed": r["seed"],
    "Cmax": r["cmax"],
    "t_comp (s)": round(r["t_comp"], 2),
    "cv": "Valid" if r["cv"] else f"Tidak valid: {r['cv_msg']}",
} for r in results])

groups = {}
for r in results:
    groups.setdefault((r["policy"], r["run_num"]), []).append(r)
summary = []
for (policy, run_num), rs in groups.items():
    s = summarize(rs)
    summary.append({
        "Method": POLICY_LABELS[policy],
        "run_num": run_num,
        "Cbest": s["c_best"],
        "Cmax avg": round(s["c_avg"], 2),
        "Cmax worst": s["c_worst"],
        "σ": round(s["sigma"], 2),
        "t_comp avg (s)": round(s["t_avg"], 2),
        "cv": f"{s['n_valid']}/{s['n_runs']}",
    })
summary_df = pd.DataFrame(summary)

tab1, tab2, tab3 = st.tabs(["Summary", "Each-Run Details", "Gantt chart"])

with tab1:
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
        x=alt.X("Method:N", title=None),
        y=alt.Y("Cmax:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Method:N", legend=None),
        column=alt.Column("run_num:O", title="run_num"),
    ).properties(width=220, height=300)
    st.altair_chart(chart)
    if not all(r["cv"] for r in results):
        st.error("Some schedules are invalid. See the Each-Run Details tab.")
    st.download_button("Download (CSV)", summary_df.to_csv(index=False), "ringkasan.csv", "text/csv")

with tab2:
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Download (CSV)", df.to_csv(index=False), "detail_run.csv", "text/csv")

with tab3:
    labels = [
        f"{POLICY_LABELS[r['policy']]} | run_num={r['run_num']} | seed={r['seed']} | Cmax={r['cmax']}"
        for r in results
    ]
    best_idx = min(range(len(results)), key=lambda k: (results[k]["cmax"], results[k]["run_num"]))
    pick = st.selectbox("Pilih jadwal", range(len(results)), index=best_idx, format_func=lambda k: labels[k])
    r = results[pick]
    sched = pd.DataFrame(r["schedule"], columns=["Job", "Step", "Mesin", "Start", "End"])
    sched["Label"] = "J" + sched["Job"].astype(str) + "-S" + sched["Step"].astype(str)
    gantt = alt.Chart(sched).mark_bar(stroke="black", strokeWidth=0.3).encode(
        x=alt.X("Start:Q", title="Waktu"),
        x2="End:Q",
        y=alt.Y("Mesin:O", title="Mesin", sort="ascending"),
        color=alt.Color("Job:N", scale=alt.Scale(scheme="tableau20")),
        tooltip=["Label", "Job", "Step", "Mesin", "Start", "End"],
    ).properties(height=max(250, 28 * problem.num_machines))
    st.altair_chart(gantt, use_container_width=True)
    st.caption(f"Cmax = {r['cmax']} | Constraint Validation: {r['cv_msg']}")
    st.download_button("Download (CSV)", sched.to_csv(index=False), "jadwal.csv", "text/csv")
