# Streamlit Prototype: Heuristic-Guided MCTS for JSSP

A prototype for running MCTS (baseline), MCTS + EDD, and MCTS + LDD on user-uploaded JSSP benchmark instances.

## Running the Application

```bash
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

## Fixed Parameters

Base seed = 0, exploration weight = 2, epsilon = 0.1, and due date factor = 1.5 (as specified in the experimental notebook).

The following parameters can be configured through the UI: method, run_num (100–3000), and NUM_RUNS (5–20).

## Project Structure

- `jssp_core.py`: Contains the parser, MCTS, JSSNode, Giffler-Thompson conflict set, EDD/LDD, and constraint validation. The logic is identical to the experimental notebook; the results for seed 0 with 1,000 iterations on ta20 are identical (2476 / 2073 / 1934).
- `app.py`: Provides the Streamlit interface (dataset input, parameter configuration, summary table, boxplot, Gantt chart, and CSV download).

## Methods

| Option | Rollout Policy |
|---|---|
| MCTS (baseline) | Random selection from all unfinished jobs |
| MCTS + EDD | G&T conflict set, earliest due date (ε-greedy) |
| MCTS + LDD | G&T conflict set, latest due date (ε-greedy) |

## Deploying to Render

The `render.yaml` file in the repository root contains the Render Blueprint configuration.

1. Push this repository to GitHub.
2. In the Render dashboard, select **New** → **Blueprint**, choose this repository, and then click **Apply**.
3. Render will execute `pip install -r requirements.txt` in the `prototype` directory, followed by:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

### Alternative Without Blueprint

Select **New** → **Web Service**, set the Root Directory to `prototype`, configure the Build Command and Start Command as described above, and set the `PYTHON_VERSION` environment variable.

**Note:** Free-tier instances go to sleep after a period of inactivity and have limited CPU resources. Therefore, experiments with large `run_num` and `NUM_RUNS` values may take a very long time to complete.