"""Inti algoritma: parser benchmark JSSP, MCTS, dan rollout policy.

Logika MCTS, JSSNode, conflict set Giffler-Thompson, EDD, LDD, dan validasi
constraint disalin dari notebook MCTS.ipynb, MCTS_EDD.ipynb, dan MCTS_LDD.ipynb.
Urutan pemanggilan `random` dijaga sama supaya hasil per seed dapat direproduksi.

Rollout policy yang tersedia:
- "random" : baseline (MCTS.ipynb), pilih job acak dari semua job yang belum selesai
- "edd"    : MCTS + EDD (MCTS_EDD.ipynb), due date terkecil di conflict set G&T
- "ldd"    : MCTS + LDD (MCTS_LDD.ipynb), due date terbesar di conflict set G&T
"""

import math
import random
import statistics
import time
from collections import defaultdict

import numpy as np

POLICY_LABELS = {
    "random": "MCTS (baseline)",
    "edd": "MCTS + EDD",
    "ldd": "MCTS + LDD",
}


# ---------------------------------------------------------------------------
# Parser instance
# ---------------------------------------------------------------------------
class JSSProblem:
    """Instance JSSP.

    Format yang didukung:
    - "taillard": baris `n m`, lalu n baris durasi, lalu n baris urutan mesin
      (indeks mesin 1-based seperti ta20.txt, atau 0-based).
    - "standard": format OR-Library, baris `n m`, lalu n baris berisi pasangan
      `mesin durasi` (indeks mesin 0-based), contoh ft06, la01, abz5.
    - "auto": deteksi otomatis dari jumlah kolom.

    Baris yang tidak seluruhnya angka (komentar, judul, "Times", "Machines")
    diabaikan. Jika baris header berisi 6 angka (format asli Taillard:
    n, m, time seed, machine seed, upper bound, lower bound), UB dan LB diambil.
    """

    def __init__(self, text, fmt="auto", due_date_factor=1.5, name="instance"):
        self.name = name
        self.warnings = []
        self.upper_bound = None
        self.lower_bound_file = None

        rows = []
        for line in text.splitlines():
            tokens = line.replace(",", " ").split()
            if not tokens:
                continue
            try:
                rows.append([int(t) for t in tokens])
            except ValueError:
                continue  # baris teks/komentar
        if not rows:
            raise ValueError("File tidak berisi data angka.")

        header = rows[0]
        if len(header) < 2:
            raise ValueError("Baris header harus berisi jumlah job dan jumlah mesin.")
        self.num_jobs, self.num_machines = header[0], header[1]
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("Jumlah job dan mesin harus positif.")
        if len(header) == 6:
            self.upper_bound, self.lower_bound_file = header[4], header[5]
        self.num_steps = self.num_machines
        data = rows[1:]
        n, m = self.num_jobs, self.num_machines

        if fmt == "auto":
            if data and len(data[0]) == 2 * m:
                fmt = "standard"
            elif len(data) >= 2 * n and all(len(r) == m for r in data[: 2 * n]):
                fmt = "taillard"
            else:
                raise ValueError(
                    "Format tidak dikenali. Taillard: 2n baris berisi m angka. "
                    "Standard: n baris berisi 2m angka (mesin durasi)."
                )
        self.format = fmt

        if fmt == "taillard":
            if len(data) < 2 * n:
                raise ValueError(f"Format Taillard butuh {2 * n} baris data, ditemukan {len(data)}.")
            durations = data[:n]
            machines = data[n: 2 * n]
            flat = [x for r in machines for x in r]
            one_based = min(flat) >= 1 and max(flat) == m
            offset = 1 if one_based else 0
            self.machine_index_base = offset
        elif fmt == "standard":
            if len(data) < n:
                raise ValueError(f"Format standard butuh {n} baris data, ditemukan {len(data)}.")
            durations, machines = [], []
            for job, r in enumerate(data[:n]):
                if len(r) != 2 * m:
                    raise ValueError(f"Job {job}: harus berisi {2 * m} angka (pasangan mesin durasi), ditemukan {len(r)}.")
                machines.append(r[0::2])
                durations.append(r[1::2])
            offset = 0
            self.machine_index_base = 0
        else:
            raise ValueError(f"Format tidak dikenal: {fmt}")

        self.jobs = []
        for job in range(n):
            if len(durations[job]) != m:
                raise ValueError(f"Job {job}: jumlah durasi {len(durations[job])} != {m}.")
            if len(machines[job]) != m:
                raise ValueError(f"Job {job}: jumlah mesin {len(machines[job])} != {m}.")
            ops = []
            for step in range(m):
                machine = machines[job][step] - offset
                duration = durations[job][step]
                if not 0 <= machine < m:
                    raise ValueError(f"Job {job}, step {step}: indeks mesin {machines[job][step]} di luar rentang.")
                if duration < 0:
                    raise ValueError(f"Job {job}, step {step}: durasi negatif.")
                ops.append((machine, duration))
            if sorted(mc for mc, _ in ops) != list(range(m)):
                self.warnings.append(f"Job {job} tidak memakai setiap mesin tepat satu kali.")
            self.jobs.append(ops)

        # Due date (metode TWK): d_j = ceil(k * total processing time job j)
        self.due_date_factor = due_date_factor
        self.due_dates = [
            math.ceil(due_date_factor * sum(d for _, d in self.jobs[job])) for job in range(n)
        ]

    @property
    def trivial_lower_bound(self):
        """LB sederhana: max(beban mesin terbesar, total durasi job terbesar)."""
        job_lb = max(sum(d for _, d in ops) for ops in self.jobs)
        load = defaultdict(int)
        for ops in self.jobs:
            for mc, d in ops:
                load[mc] += d
        return max(job_lb, max(load.values()))


# ---------------------------------------------------------------------------
# MCTS (sama dengan notebook; rollout memilih policy)
# ---------------------------------------------------------------------------
class MCTS:
    def __init__(self, exploration_weight=2, epsilon=0.1, policy="random"):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon
        self.policy = policy
        self.best_reward = float("-inf")
        self.best_node = None

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagation(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._ucb_select(node)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _rollout_child(self, node):
        if self.policy == "random":
            return node.find_random_child()
        return node.find_heuristic_child(self.epsilon, self.policy)

    def _simulate(self, node):
        while True:
            if node.is_terminal():
                reward = node.reward()
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_node = node
                return reward
            child = self._rollout_child(node)
            if child is None:
                reward = node.reward()
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_node = node
                return reward
            node = child

    def _backpropagation(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _ucb_select(self, node):
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def ucb(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=ucb)


class JSSNode:
    __slots__ = ("problem", "job_steps", "job_completion_times", "machine_available_times", "parent", "operation")

    def __init__(self, problem, job_steps, job_completion_times, machine_available_times, parent=None, operation=None):
        self.problem = problem
        self.job_steps = tuple(job_steps)
        self.job_completion_times = tuple(job_completion_times)
        self.machine_available_times = tuple(machine_available_times)
        self.parent = parent
        self.operation = operation

    def _make_child(self, job):
        current_step = self.job_steps[job]
        machine, duration = self.problem.jobs[job][current_step]
        start_time = max(self.job_completion_times[job], self.machine_available_times[machine])
        end_time = start_time + duration
        new_job_steps = list(self.job_steps)
        new_job_steps[job] += 1
        new_job_completion = list(self.job_completion_times)
        new_job_completion[job] = end_time
        new_machine_available = list(self.machine_available_times)
        new_machine_available[machine] = end_time
        return JSSNode(self.problem, new_job_steps, new_job_completion, new_machine_available, self,
                       (job, current_step, machine, start_time, end_time))

    def find_children(self):
        if self.is_terminal():
            return set()
        children = set()
        for job in range(self.problem.num_jobs):
            if self.job_steps[job] >= self.problem.num_steps:
                continue
            children.add(self._make_child(job))
        return children

    def find_random_child(self):
        if self.is_terminal():
            return None
        possible_jobs = [j for j in range(self.problem.num_jobs) if self.job_steps[j] < self.problem.num_steps]
        if not possible_jobs:
            return None
        job = random.choice(possible_jobs)
        return self._make_child(job)

    def conflict_set(self):
        """Conflict set Giffler-Thompson."""
        earliest_start = {}
        earliest_completion = {}
        for job in range(self.problem.num_jobs):
            current_step = self.job_steps[job]
            if current_step >= self.problem.num_steps:
                continue
            machine, duration = self.problem.jobs[job][current_step]
            es = max(self.job_completion_times[job], self.machine_available_times[machine])
            earliest_start[job] = es
            earliest_completion[job] = es + duration
        if not earliest_start:
            return []
        j_star = min(earliest_completion, key=lambda j: (earliest_completion[j], j))
        m_star = self.problem.jobs[j_star][self.job_steps[j_star]][0]
        ec_star = earliest_completion[j_star]
        return [
            job for job in earliest_start
            if self.problem.jobs[job][self.job_steps[job]][0] == m_star and earliest_start[job] < ec_star
        ]

    def find_heuristic_child(self, epsilon=0.0, rule="edd"):
        if self.is_terminal():
            return None
        candidates = self.conflict_set()
        if not candidates:
            return None
        if random.random() < epsilon:
            job = random.choice(candidates)  # eksplorasi
        else:
            due = self.problem.due_dates
            if rule == "edd":
                job = min(candidates, key=lambda j: (due[j], j))
            elif rule == "ldd":
                job = max(candidates, key=lambda j: (due[j], -j))
            else:
                raise ValueError(f"Aturan tidak dikenal: {rule}")
        return self._make_child(job)

    def is_terminal(self):
        return all(step == self.problem.num_steps for step in self.job_steps)

    def reward(self):
        return -max(self.job_completion_times)

    def __hash__(self):
        return hash((self.job_steps, self.job_completion_times, self.machine_available_times))

    def __eq__(self, other):
        return (
            self.job_steps == other.job_steps
            and self.job_completion_times == other.job_completion_times
            and self.machine_available_times == other.machine_available_times
        )

    def __repr__(self):
        return f"JSSNode(makespan={max(self.job_completion_times)})"


# ---------------------------------------------------------------------------
# Utilitas jadwal & eksperimen
# ---------------------------------------------------------------------------
def get_schedule(node):
    schedule = []
    current = node
    while current.parent is not None:
        if current.operation:
            schedule.append(current.operation)
        current = current.parent
    schedule.reverse()
    return schedule


def check_constraints(schedule, problem):
    """Validasi jadwal. Return (valid, pesan)."""
    expected = problem.num_jobs * problem.num_steps
    if len(schedule) != expected:
        return False, f"Jumlah operasi {len(schedule)} != {expected}"
    ops = {}
    for job, step, machine, start, end in schedule:
        if (job, step) in ops:
            return False, f"Operasi J{job}-S{step} dijadwalkan lebih dari sekali"
        exp_machine, exp_duration = problem.jobs[job][step]
        if machine != exp_machine or end - start != exp_duration or start < 0:
            return False, f"Operasi J{job}-S{step} tidak sesuai data instance"
        ops[(job, step)] = (machine, start, end)
    for job in range(problem.num_jobs):
        for step in range(1, problem.num_steps):
            if ops[(job, step)][1] < ops[(job, step - 1)][2]:
                return False, f"Precedence dilanggar pada job {job}, step {step}"
    by_machine = defaultdict(list)
    for machine, start, end in ops.values():
        by_machine[machine].append((start, end))
    for machine, intervals in by_machine.items():
        intervals.sort()
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            if s2 < e1:
                return False, f"Overlap pada mesin {machine}: [{s1},{e1}) dan [{s2},{e2})"
    return True, "Valid"


def run_single(problem, run_num, seed, exploration_weight, epsilon, policy, progress_cb=None):
    """Satu run MCTS. Return dict berisi Cmax, cv, t_comp, dan jadwal terbaik."""
    random.seed(seed)
    np.random.seed(seed)
    root = JSSNode(problem, [0] * problem.num_jobs, [0] * problem.num_jobs, [0] * problem.num_machines)
    mcts = MCTS(exploration_weight=exploration_weight, epsilon=epsilon, policy=policy)

    step = max(1, run_num // 50)
    t_start = time.perf_counter()
    for i in range(run_num):
        mcts.do_rollout(root)
        if progress_cb and (i + 1) % step == 0:
            progress_cb((i + 1) / run_num)
    t_comp = time.perf_counter() - t_start

    best = mcts.best_node
    cmax = -best.reward()
    schedule = get_schedule(best)
    cv, cv_msg = check_constraints(schedule, problem)
    if cv and max(end for *_, end in schedule) != cmax:
        cv, cv_msg = False, "Makespan jadwal tidak sama dengan makespan node"
    return {
        "policy": policy, "seed": seed, "run_num": run_num, "cmax": cmax,
        "cv": cv, "cv_msg": cv_msg, "t_comp": t_comp, "schedule": schedule,
    }


def summarize(results):
    cmaxs = [r["cmax"] for r in results]
    times = [r["t_comp"] for r in results]
    return {
        "policy": results[0]["policy"],
        "run_num": results[0]["run_num"],
        "n_runs": len(results),
        "c_best": min(cmaxs),
        "c_avg": statistics.mean(cmaxs),
        "c_worst": max(cmaxs),
        "sigma": statistics.stdev(cmaxs) if len(cmaxs) > 1 else 0.0,
        "n_valid": sum(r["cv"] for r in results),
        "t_avg": statistics.mean(times),
        "t_total": sum(times),
    }
