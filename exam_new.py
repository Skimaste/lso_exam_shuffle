import numpy as np 
from time import time
import json

from KPdp import KPsolver # provided integer knapsack solver
from data_reader import DataReader # for reading datasets
from feasible import ubs # reported upper bounds (just a list of integers)


class exam_solver:
    def __init__(self, max_iter: int = 1000, warm: bool = True, solution: bool = False):
        """
        Solve the exam-room assignment problem using Lagrangian relaxation
        and subgradient optimization.
        """
        self.max_iter = max_iter
        self.dataset = None

        self.n = 0
        self.m = 0
        self.h = None
        self.r = None
        self.p = None
        self.c = None

        self.lb_best = -np.inf
        self.reported_ub = np.inf
        self.best_ub = np.inf

        self.warm = warm
        self.solution = solution

        self.gap = np.inf
        self.iterations = 0
        self.time_sec = 0.0
        self.stop_reason = "not started"

        # Latest relaxed iterate
        self.x = None
        self.y = None

        # Best feasible primal solution found
        self.x_feas = None
        self.y_feas = None

    def __kpsolve__(self, profits, weights, capacity):
        """Wrapper for the provided integer knapsack solver."""
        kpsolver = KPsolver()
        kpsolver.setData(len(profits), profits, weights, capacity)
        return kpsolver.solve_knapsack()

    def __read_data__(self):
        """Read one dataset and return (data, reported upper bound)."""
        reader = DataReader()
        data = reader.read_data_from_file(f"./All_data_sets/p{self.dataset}.txt")
        return data, int(ubs[self.dataset - 1])

    def __true_cost__(self, x, y):
        return int(np.sum(self.h * x) + np.sum(self.r * y))

    def __check_feasibility__(self, x, y):
        """
        Feasibility check for the exactly-one assignment model:
        - each room capacity respected
        - each exam assigned to exactly one room
        - assignments only allowed in open rooms
        """
        if x is None or y is None:
            return False

        # Capacity constraints
        for i in range(self.n):
            assigned_exams = np.where(x[i] == 1)[0]
            total_space = np.sum(self.p[assigned_exams])
            if total_space > self.c[i] * y[i]:
                return False

        # Exactly-one assignment constraints
        for j in range(self.m):
            assigned_count = int(np.sum(x[:, j]))
            if assigned_count != 1:
                return False

        # Exams only in open rooms
        for i in range(self.n):
            if y[i] == 0 and np.any(x[i] == 1):
                return False

        return True

    def __incremental_cost__(self, room_idx, exam_idx, y):
        """True objective delta for assigning exam j to room i."""
        open_cost = self.r[room_idx] if y[room_idx] == 0 else 0
        return self.h[room_idx, exam_idx] + open_cost

    def __local_improve__(self, x, y, remaining, max_rounds=3):
        """
        Simple local search: single-exam relocate moves that reduce true cost.
        Includes room-close effect when moving the last exam out of a room.
        """
        for _ in range(max_rounds):
            improved = False
            loads = self.c - remaining

            for j in range(self.m):
                src = int(np.where(x[:, j] == 1)[0][0])
                current_delta = self.h[src, j]

                # If exam j is the only one in src, moving it can close the room.
                if loads[src] == self.p[j]:
                    current_delta += self.r[src]

                best_room = src
                best_delta = current_delta

                for dst in range(self.n):
                    if dst == src:
                        continue
                    if remaining[dst] < self.p[j]:
                        continue

                    cand_delta = self.h[dst, j] + (self.r[dst] if y[dst] == 0 else 0)
                    if cand_delta < best_delta:
                        best_delta = cand_delta
                        best_room = dst

                if best_room != src:
                    # Move exam j from src to best_room
                    x[src, j] = 0
                    remaining[src] += self.p[j]

                    x[best_room, j] = 1
                    remaining[best_room] -= self.p[j]
                    y[best_room] = 1

                    # Close src if empty
                    if np.all(x[src] == 0):
                        y[src] = 0

                    loads = self.c - remaining
                    improved = True

            if not improved:
                break

        return x, y

    def __primal_recovery__(self, x_star):
        """
        Build an exactly-one feasible solution from relaxed x_star using true costs.

        1) Seed with room proposals from x_star (keep cheapest feasible room)
        2) Greedy insert for unassigned exams by true incremental cost
        3) Local improve via single-exam relocation moves
        """
        n, m = self.n, self.m
        x = np.zeros((n, m), dtype=int)
        y = np.zeros(n, dtype=int)
        remaining = self.c.astype(float).copy()
        assigned = np.zeros(m, dtype=bool)

        # Step 1: keep best candidate room among rooms selected in x_star.
        room_counts = x_star.sum(axis=0)
        exam_order = np.lexsort((-self.p, -room_counts))

        for j in exam_order:
            candidate_rooms = np.where(x_star[:, j] == 1)[0]
            if len(candidate_rooms) == 0:
                continue

            best_room = -1
            best_cost = np.inf
            for i in candidate_rooms:
                if remaining[i] >= self.p[j]:
                    cst = self.__incremental_cost__(i, j, y)
                    if cst < best_cost:
                        best_cost = cst
                        best_room = i

            if best_room >= 0:
                x[best_room, j] = 1
                y[best_room] = 1
                remaining[best_room] -= self.p[j]
                assigned[j] = True

        # Step 2: assign all remaining exams by best true incremental cost.
        unassigned = np.where(~assigned)[0]
        unassigned = unassigned[np.argsort(-self.p[unassigned])]

        for j in unassigned:
            best_room = -1
            best_cost = np.inf
            for i in range(n):
                if remaining[i] >= self.p[j]:
                    cst = self.__incremental_cost__(i, j, y)
                    if cst < best_cost:
                        best_cost = cst
                        best_room = i

            if best_room >= 0:
                x[best_room, j] = 1
                y[best_room] = 1
                remaining[best_room] -= self.p[j]

        # Local search polishing on true objective
        x, y = self.__local_improve__(x, y, remaining)

        feas = self.__check_feasibility__(x, y)
        cost = self.__true_cost__(x, y)
        return x, y, cost, feas

    def __solve__(self):
        """Internal Lagrangian + subgradient optimization loop."""
        if self.warm:
            mu = np.min(self.h, axis=0).astype(float) # Got this from copilot
        else:
            mu = np.zeros(self.m, dtype=float)

        lam = 2.0
        no_improvement = 0
        self.stop_reason = "ran out of iterations"

        for k in range(self.max_iter):
            # Solve decomposed knapsack subproblems for current multipliers
            self.x = np.zeros((self.n, self.m), dtype=int)
            self.y = np.zeros(self.n, dtype=int)
            total = float(np.sum(mu))

            for i in range(self.n):
                profits = np.maximum(0.0, mu - self.h[i])
                kp_val, selected = self.__kpsolve__(profits.astype(int), self.p, self.c[i])

                if self.r[i] - kp_val < 0:
                    self.y[i] = 1
                    self.x[i, selected] = 1
                    total += self.r[i] - kp_val

            lb = int(np.ceil(total))
            if lb > self.lb_best:
                self.lb_best = lb
                no_improvement = 0
            else:
                no_improvement += 1

            # Recover primal every iteration and keep only best feasible UB.
            x_cand, y_cand, ub_candidate, is_feas = self.__primal_recovery__(self.x)
            if is_feas and ub_candidate < self.best_ub:
                self.best_ub = int(ub_candidate)
                self.x_feas = x_cand.copy()
                self.y_feas = y_cand.copy()

            # Subgradient for relaxed constraints sum_i x_ij >= 1
            s = 1 - self.x.sum(axis=0)
            norm_sq = float(np.dot(s, s))
            if norm_sq == 0:
                self.stop_reason = "Optimal solution found (subgradient is zero)"
                self.iterations = k + 1
                return

            # Use feasible UB when available, else fallback to reported UB.
            ub_for_step = self.best_ub if np.isfinite(self.best_ub) else self.reported_ub
            alpha = lam * (ub_for_step - lb) / norm_sq
            mu += alpha * s

            if no_improvement >= 30:
                lam /= 2.0
                no_improvement = 0
                if lam < 1e-12:
                    self.stop_reason = "Step size multiplier too small, stopping optimization"
                    self.iterations = k + 1
                    return

        self.iterations = self.max_iter

    def solve(self, dataset: int) -> dict:
        """
        Solve one dataset and return bounds, gap, timing, and optional solutions.
        """
        self.dataset = dataset

        data, self.reported_ub = self.__read_data__()
        self.n = int(data["n"])
        self.m = int(data["m"])
        self.h = np.array(data["h"], dtype=int)
        self.r = np.array(data["r"], dtype=int)
        self.p = np.array(data["p"], dtype=int)
        self.c = np.array(data["c"], dtype=int)

        # Reset per-instance state
        self.lb_best = -np.inf
        self.best_ub = np.inf
        self.x_feas = None
        self.y_feas = None

        start_time = time()
        self.__solve__()
        self.time_sec = time() - start_time

        if np.isfinite(self.best_ub) and np.isfinite(self.lb_best):
            self.gap = (self.best_ub - self.lb_best) / (self.lb_best + 1e-8) * 100
        else:
            self.gap = np.inf

        return {
            "dataset": self.dataset,
            "reported_ub": self.reported_ub,
            "found_ub": int(self.best_ub) if np.isfinite(self.best_ub) else None,
            "lb": int(self.lb_best) if np.isfinite(self.lb_best) else None,
            "gap": float(self.gap),
            "iterations": int(self.iterations),
            "time_sec": float(self.time_sec),
            "feasible": self.__check_feasibility__(self.x_feas, self.y_feas),
            "stop_reason": self.stop_reason,
            "solution": (self.x, self.y) if self.solution else None,
            "feasible_solution": (self.x_feas, self.y_feas) if self.solution else None,
        }

    def all(self) -> list[dict]:
        """Solve all 45 datasets."""
        results = []
        for i in range(1, 46):
            print(f"Solving dataset {i}...")
            results.append(self.solve(dataset=i))
            print(f"Dataset {i} done. LB: {results[-1]['lb']}, UB: {results[-1]['found_ub']}, Gap: {results[-1]['gap']:.2f}%, Time: {results[-1]['time_sec']:.2f}s, Stop reason: {results[-1]['stop_reason']}")
        return results

    def solution_printer(self, x, y, reported_ub=None): 
        if x is None or y is None:
            print("No solution available.")
            return

        print(np.sum(x, axis=0))
        cost = self.__true_cost__(x, y)
        if reported_ub is None:
            reported_ub = self.reported_ub
        print(f"Calculated cost of the solution: {cost}, reported upper bound: {reported_ub}")

        for i in range(self.n):
            assigned_exams = np.where(x[i] == 1)[0]
            total_space = np.sum(self.p[assigned_exams])
            if total_space > self.c[i] * y[i]:
                print(
                    f"Capacity violated for room {i}: load {total_space} > cap {self.c[i]} with y={y[i]}"
                )

        for j in range(self.m):
            assigned_rooms = np.where(x[:, j] == 1)[0]
            if len(assigned_rooms) != 1:
                print(
                    f"Assignment violated for exam {j}: assigned to rooms {assigned_rooms} (must be exactly one)"
                )

        for i in range(self.n):
            if y[i] == 0 and np.any(x[i] == 1):
                print(f"Open-room violated: room {i} is closed but has assigned exams")

    def __to_serializable__(self, obj): # vibe coded
        """Recursively convert NumPy-heavy objects into JSON-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self.__to_serializable__(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.__to_serializable__(v) for v in obj]
        return obj

    def save_result( # vibe coded as well
        self,
        result: dict,
        file_path: str,
        include_solutions: bool = False,
        pretty: bool = False,
    ):
        """
        Save one result dictionary to JSON.

        By default this stores only scalar metrics. Set include_solutions=True
        to also include x/y matrices, which can make files very large.
        Set pretty=True for indented output. Compact mode keeps arrays on one line.
        """
        payload = dict(result)
        if not include_solutions:
            payload.pop("solution", None)
            payload.pop("feasible_solution", None)

        with open(file_path, "w") as f:
            if pretty:
                json.dump(self.__to_serializable__(payload), f, indent=2)
            else:
                json.dump(self.__to_serializable__(payload), f, separators=(",", ":"))

    def save_all_results( # vibe coded as well
        self,
        results: list[dict],
        file_path: str,
        include_solutions: bool = False,
        pretty: bool = False,
    ):
        """
        Save per-dataset result dictionaries as JSON Lines (one dataset per line).

        By default this stores only scalar metrics. Set include_solutions=True
        to also include x/y matrices for each dataset.
        """
        with open(file_path, "w") as f:
            for res in results:
                compact = dict(res)
                if not include_solutions:
                    compact.pop("solution", None)
                    compact.pop("feasible_solution", None)

                serializable = self.__to_serializable__(compact)
                if pretty:
                    line = json.dumps(serializable, indent=2)
                else:
                    line = json.dumps(serializable, separators=(",", ":"))
                f.write(line + "\n")


if __name__ == "__main__":
    solver = exam_solver(max_iter=3000, solution=True)
    result = solver.all()

    # Save compact metrics to JSON (without x/y arrays)
    # solver.save_result(result, "result_p2.json", include_solutions=True)
    solver.save_all_results(result, "all_results.json", include_solutions=False, pretty=False)
    solver.save_all_results(result, "all_results_with_solutions.json", include_solutions=True, pretty=False)


    # print(result)
    # 
    # print("\n\n\nChecking feasibility and calculating cost of the relaxed solution:")
    # x_rel, y_rel = result["solution"]
    # solver.solution_printer(x_rel, y_rel, reported_ub=result["reported_ub"])
    # 
    # print("\n\n\nChecking feasibility and calculating cost of the primal recovery solution:")
    # x_feas, y_feas = result["feasible_solution"]
    # solver.solution_printer(x_feas, y_feas, reported_ub=result["reported_ub"])
