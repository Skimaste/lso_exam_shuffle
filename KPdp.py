import numpy as np


class KPsolver:
    """
    This class implements a simple dynamic programming algorithm for solving 0-1 knapsack problems
    """
    def __init__(self):
        """
        Initialize the internal variables and data-structure
        """
        self.n = 0  # Number of items in the knapsack problem
        self.profits = []  # List of profits for each item
        self.weights = []  # List of weights for each item
        self.capacity = 0  # Integer specifying the capacity of the knapsack
        self.optValue = 0  # Optimal objective function value
        self.zeroOneSolved = None  # Flag which is True if the 0-1 KP is solved to optimality
        self.free = []

    def setData(self, n: int, profits: list, weights: list, capacity: int):
        """
        This method initializes the problem parameters including the number of items,
        their associated profits and weights, and the total capacity of the knapsack.
        Calling this method resets the solver's state, clearing any existing dynamic
        programming tables and solution flags.

        Parameters:
            n: Integer specifying the number of items in the knapsack
            profits: List of profits for each item. MUST BE INTEGERS/NUMPY INTEGERS.
            weights: List of weights for each item. MUST BE INTEGERS/NUMPY INTEGERS.
            capacity: Integer specifying the capacity of the knapsack

         Note:
             This method overwrites any previously set data in the KPsolver instance.
             Input must be integers. If not, a TypeError is raised
        """
        self.n = n
        self.profits = np.array(profits)  # Convert to numpy array
        self.weights = np.array(weights)  # Convert to numpy array
        if not np.issubdtype(self.profits.dtype, np.integer):
            raise TypeError("All profit values must be integers.")
        if not np.issubdtype(self.weights.dtype, np.integer):
            raise TypeError("All weight values must be integers.")

        self.capacity = capacity
        self.zeroOneSolved = False  # Assume the 0-1 KP is not solved to optimality

    def solve_knapsack(self):
        """
        Solves the 0-1 knapsack problem using a vectorized NumPy dynamic programming approach.

        This method implements the standard $O(nC)$ algorithm but leverages NumPy
        slicing to execute the inner capacity loop at C-speeds. It computes both
        the maximum possible value and the specific indices of the items included
        in the optimal set.

        Note:
            This method requires `setData` to be called first to initialize
            `self.values`, `self.weights`, and `self.capacity`.

        Returns:
            tuple: A tuple containing:
                - int: The maximum total value achievable within the capacity.
                - list[int]: A sorted list of indices representing the items
                  included in the optimal solution.

        Raises:
            AttributeError: If `setData` has not been called or the required
                attributes are missing.
            MemoryError: If the capacity and number of items result in an
                array size that exceeds available RAM.

        Complexity:
            Time: $O(n x C)$ where $n$ is the number of items and $C$ is the capacity.
            Space: $O(n x C)$ to store the 2D DP table for solution reconstruction.
        """
        # --- Separate items with positive profits from those trivially fixed to zero ---
        positive_indices = [i for i in range(len(self.profits)) if self.profits[i] > 0]

        # If no items have positive profit, the optimal solution is trivially zero
        if not positive_indices:
            self.optValue = 0
            self.zeroOneSolved = True
            return 0, []

        reduced_profits = self.profits[positive_indices]
        reduced_weights = self.weights[positive_indices]
        n_reduced = len(positive_indices)

        # --- Build DP table over the reduced item set only ---
        dp = np.zeros((n_reduced + 1, self.capacity + 1), dtype=np.int64)

        for i in range(1, n_reduced + 1):
            v, w = reduced_profits[i - 1], reduced_weights[i - 1]
            dp[i, :] = dp[i - 1, :]
            if w <= self.capacity:
                take_value = v + dp[i - 1, :-w]
                dp[i, w:] = np.maximum(dp[i, w:], take_value)

        # --- Backtrack over the reduced table to find included items ---
        included_reduced = []
        curr_w = self.capacity
        for i in range(n_reduced, 0, -1):
            if dp[i, curr_w] != dp[i - 1, curr_w]:
                included_reduced.append(i - 1)
                curr_w -= reduced_weights[i - 1]

        # --- Map reduced indices back to original indices ---
        included_indices = [positive_indices[i] for i in included_reduced]

        self.optValue = dp[n_reduced, self.capacity]
        self.zeroOneSolved = True
        return self.optValue, included_indices

    def solveFractionalKP(self):
        """
        Solves the Linear Programming (LP) relaxation of the 0-1 Knapsack Problem.

        Uses the greedy 'Dantzig' algorithm: items are sorted by their profit-to-weight
        ratio ($p_i / w_i$) and added to the knapsack in non-increasing order. When an
        item cannot fit entirely, the remaining capacity is filled with a fraction
        of that item (the 'break item').

        The optimal value of this relaxation provides a simple upper bound
        for the 0-1 Knapsack Problem.

        Returns:
            list[float]: A list of length `n` where each entry x_i in [0, 1]
                represents the fraction of item $i$ included in the optimal
                LP solution.

        Note:
            - The total value of the solution is stored in `self.optValue`.
            - If the total weight of all items is less than `self.capacity`,
              all $x_i$ will be 1.0.
            - Time Complexity: $O(n log( n ) )$ due to sorting.
            - Space Complexity: $O(n)$ to store the solution vector and sorted indices.

        Example:
            If item profits are [10, 10, 20]  and the weights are [1, 2, 10] and the capacity is 2
            We have the profit-to-weight ratios : [10, 5, 2]. Hence, we start by adding item 1, x[1]=1  then half
            of item 2, x[2]=0.5, which exhausts the capacity of the knapsack. The solution value is then 15.
        """
        # Vectorized ratio calculation and sorting
        ratios = self.profits / self.weights
        indices_sorted = np.argsort(ratios)[::-1]  # Sort descending

        x = np.zeros(self.n)
        cumulative_weight = 0
        self.optValue = 0.0

        for idx in indices_sorted:
            if cumulative_weight + w[idx] <= self.capacity:
                x[idx] = 1.0
                cumulative_weight += w[idx]
                self.optValue += p[idx]
            else:
                # The 'break item'
                remaining = self.capacity - cumulative_weight
                x[idx] = remaining / w[idx]
                self.optValue += p[idx] * x[idx]
                break

        return self.optValue, x


if __name__ == '__main__':
    import time

    kpsolver = KPsolver()

    p = [27, 18, 43, 40, 55, 15, 10, 91, 90, 95]#, 5, 66, 59, 1, 94, 62, 43, 20, 52, 32, 29, 40, 58, 75, 84, 94, 8, 1, 63, 66, 75, 26, 33, 56, 42, 16, 81, 89, 44, 95, 95, 52, 58, 75, 38, 20, 11, 75, 12, 42, 8, 76, 20, 87, 22, 72, 98, 17, 25, 32, 92, 16, 17, 24, 49, 53, 44, 8, 30, 12, 6, 85, 13, 51, 98, 55, 76, 36, 38, 65, 53, 16, 31, 7, 1, 94, 68, 14, 16, 39, 42, 60, 54, 76, 74, 86, 25, 69, 18, 62, 32, 69, 18, 22, 91, 27, 90, 37, 23, 54, 24, 83, 1, 64, 97, 64, 70, 5, 43, 41, 4, 1, 20, 57, 93, 67, 45, 68, 20, 86, 29, 12, 71, 95, 4, 47, 95, 33, 18, 58, 85, 28, 54, 11, 25, 91, 54, 85, 67, 2, 20, 34, 34, 39, 57, 74, 13, 23, 7, 14, 44, 38, 84, 20, 34, 42, 25, 73, 15, 13, 22, 11, 43, 66, 91, 34, 65, 94, 44, 18, 37, 63, 12, 100, 27, 92, 49, 67, 81, 91, 62, 8, 83, 37, 14, 61, 3, 45, 12, 32, 39, 68, 89, 71, 74, 58, 24, 16, 98, 6, 52, 52, 5, 96, 44, 42, 49, 29, 28, 92, 61, 73, 100, 75, 39, 42, 85, 52, 79, 10, 32, 88, 64, 42, 30, 41, 1, 95, 24, 44, 66, 77, 25, 92, 91, 31, 42, 12, 5, 48, 70, 61, 65, 26, 82, 83, 8, 76, 4, 84, 34, 17, 78, 37, 8, 95, 13, 12, 28, 93, 98, 12, 25, 48, 77, 40, 3, 51, 6, 89, 91, 30, 7, 77, 65, 34, 50, 29, 100, 75, 35, 12, 10, 80, 83, 61, 5, 83, 80, 25, 86, 57, 3, 8, 64, 22, 40, 38, 34, 27, 18, 33, 53, 67, 86, 50, 12, 40, 25, 74, 7, 70, 4, 31, 7, 68, 23, 8, 35, 34, 72, 96, 53, 53, 74, 15, 29, 19, 67, 26, 34, 70, 79, 65, 1, 41, 39, 71, 49, 32, 52, 19, 25, 75, 94, 46, 50, 41, 76, 18, 56, 24, 26, 56, 21, 18, 30, 53, 81, 56, 6, 1, 48, 65, 6, 63, 81, 75, 68, 76, 78, 58, 73, 29, 45, 13, 79, 56, 50, 4, 3, 29, 17, 40, 88, 15, 22, 40, 21, 88, 27, 54, 15, 54, 25, 13, 42, 66, 22, 29, 97, 36, 94, 13, 67, 74, 5, 15, 12, 15, 38, 80, 82, 41, 75, 40, 32, 24, 54, 47, 76, 50, 10, 19, 99, 26, 8, 61, 40, 70, 91, 17, 57, 5, 58, 22, 94, 23, 16, 22, 49, 59, 89, 30, 59, 47, 66, 29, 61, 85, 53, 72, 4, 83, 50, 54, 49, 48, 21, 17, 7, 60, 91, 98, 38, 47, 58, 40, 95, 75, 96, 7, 19, 15, 13, 33, 95, 28, 45, 73, 63, 40, 86, 73, 87, 46, 38, 8, 14, 32]
    w = [20, 2, 16, 30, 19, 42, 2, 21, 13, 2]#, 2, 29, 49, 34, 26, 4, 25, 45, 11, 29, 20, 17, 8, 31, 36, 16, 13, 4, 24, 37, 4, 31, 49, 15, 33, 11, 22, 24, 42, 16, 39, 32, 24, 42, 47, 45, 33, 33, 27, 3, 7, 10, 39, 5, 30, 13, 10, 15, 11, 43, 33, 18, 22, 5, 40, 40, 11, 7, 36, 16, 45, 14, 42, 46, 43, 11, 45, 36, 49, 10, 14, 34, 11, 36, 20, 49, 35, 50, 2, 18, 26, 33, 33, 8, 10, 33, 48, 15, 43, 11, 8, 14, 42, 50, 37, 11, 42, 49, 30, 48, 27, 9, 48, 29, 36, 24, 21, 43, 14, 23, 50, 42, 31, 18, 27, 38, 36, 16, 15, 49, 35, 7, 28, 45, 35, 13, 29, 17, 31, 45, 41, 20, 14, 13, 42, 8, 36, 48, 18, 3, 40, 45, 44, 16, 27, 6, 11, 21, 48, 50, 11, 48, 46, 3, 44, 1, 10, 22, 27, 12, 14, 25, 7, 48, 26, 2, 44, 25, 6, 8, 5, 35, 22, 45, 14, 6, 33, 4, 43, 31, 48, 25, 40, 9, 47, 7, 10, 5, 46, 39, 15, 50, 17, 13, 37, 47, 28, 7, 15, 39, 44, 14, 34, 30, 48, 34, 21, 14, 25, 33, 44, 19, 8, 15, 25, 1, 31, 13, 20, 32, 21, 29, 4, 37, 29, 19, 36, 47, 48, 12, 3, 3, 5, 16, 39, 45, 50, 30, 7, 14, 31, 9, 9, 28, 14, 12, 48, 13, 13, 6, 40, 14, 42, 26, 41, 12, 39, 42, 9, 35, 29, 48, 25, 6, 14, 20, 38, 28, 26, 48, 15, 39, 43, 6, 29, 47, 31, 47, 34, 44, 13, 1, 43, 10, 14, 1, 42, 38, 7, 33, 35, 47, 16, 16, 4, 31, 1, 38, 8, 28, 43, 17, 48, 24, 31, 15, 26, 13, 36, 7, 1, 34, 17, 40, 2, 28, 19, 15, 5, 36, 4, 30, 38, 1, 13, 11, 29, 20, 24, 17, 4, 18, 14, 5, 1, 23, 2, 40, 29, 42, 19, 49, 5, 44, 8, 15, 1, 35, 9, 14, 44, 19, 17, 25, 19, 9, 21, 32, 25, 22, 1, 9, 4, 12, 7, 4, 50, 24, 7, 41, 36, 19, 50, 31, 8, 48, 48, 50, 43, 7, 5, 17, 43, 22, 26, 39, 15, 39, 34, 38, 44, 2, 15, 23, 10, 41, 28, 42, 9, 12, 21, 35, 22, 27, 36, 15, 22, 37, 24, 28, 12, 20, 10, 9, 43, 41, 23, 8, 27, 9, 45, 22, 3, 43, 43, 24, 5, 16, 38, 41, 36, 21, 38, 44, 13, 25, 31, 2, 35, 2, 42, 14, 21, 3, 9, 35, 16, 39, 6, 4, 38, 28, 34, 12, 19, 17, 46, 37, 35, 30, 36, 10, 7, 25, 35, 7, 6, 33, 42, 48, 41, 18, 18, 13, 25, 40, 27, 34, 49, 19, 40, 49, 38, 32, 30, 49, 6, 7, 20, 40]
    cap = 80#8792

    kpsolver.setData(len(p), p, w, cap)

    start = time.time()
    opt_value, opt_solution = kpsolver.solve_knapsack()
    print(f'Total time for DP-algorithm is {time.time() - start} seconds')
    print (f'Optimal solution value is {opt_value}')
    print(f'Optimal solution consists of items: {opt_solution}')

    start = time.time()
    opt_value, opt_solution = kpsolver.solveFractionalKP()
    print(f'Total time for fractional DP is {time.time() - start} seconds')
    print(f'Optimal fractional solution value is {opt_value}')
    print(f'Optimal solution is given by: {opt_solution}')
