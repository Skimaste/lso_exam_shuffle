# This file implements a reader for reading data sets for the exam-room assignment problem studied in the
# Large Scale Optimisation course, at Aarhus University, spring 2026.
#
# Author: Sune Lauth Gadegaard

class DataReader:
    def __init__(self):
        """ Initialises internal variables """
        self.n = 0  # Number of rooms
        self.m = 0  # Number of exams
        self.h = []  # List of lists with "inconvenience measure"
        self.r = []  # List of room costs
        self.c = []  # List of room capacities
        self.p = []  # List of exam space requirements

    def read_data_from_file(self, file_name: str) -> dict:
        """
        Reads a exam assignment instance from a text file into a dictionary format

        File format:
            Line 1:             n  m          (number of rooms, number of exams)
            Next n lines:       c[i]  r[i]    (capacity and room opening cost for room i)
            Next line:          p[0] p[1] ... p[m-1]   (space requirements for each exam, space-separated)
            Next n lines:       h[i][0] h[i][1] ... h[i][m-1]  (inconvenience measure, row per room)

            Blank lines between sections are ignored.

        Returns a dict with keys:
            'n' : int              - number of rooms
            'm' : int              - number of exams
            'c' : list[int]        - room capacities,        length n
            'r' : list[int]        - room costs,             length n
            'p' : list[int]        - exam requirements,      length m
            'h' : list[list[int]]  - inconvenience measure,  n x m  (h[i][j] = inconvenience of assigning exam j to room i)
        """
        with open(file_name) as file:
            # Read all non-empty lines, stripping whitespace
            lines = [line.strip() for line in file if line.strip()]

        token_iter = iter(lines)

        # --- n and m ---
        first_line = next(token_iter).split()
        n, m = int(first_line[0]), int(first_line[1])

        # --- n lines of (capacity, fixed_cost) pairs ---
        c = []  # capacities
        r = []  # fixed costs
        for _ in range(n):
            parts = next(token_iter).split()
            c.append(int(parts[0]))
            r.append(int(parts[1]))

        # --- demand line ---
        req_tokens = next(token_iter).split()
        p = [int(v) for v in req_tokens]

        if len(p) != m:
            raise ValueError(f"Expected {m} exam requirements, but read {len(p)}.")

        # --- n lines of assignment costs ---
        h = []
        for i in range(n):
            inc_tokens = next(token_iter).split()
            row = [int(v) for v in inc_tokens]
            if len(row) != m:
                raise ValueError(f"Expected {m} inconvenience values on row {i}, but read {len(row)}.")
            h.append(row)

        return {'n': n, 'm': m, 'c': c, 'r': r, 'p': p, 'h': h}


# if __name__ == '__main__':
#     reader = DataReader()
#     print(reader.read_data_from_file("./100-500/R7/p45.txt"))