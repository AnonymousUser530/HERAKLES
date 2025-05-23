# taken from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
# and modified to work with our settings
import numpy as np
from collections import deque

class SumTree:
    write = 0

    def __init__(self, queue: deque, max_priority=1.0):
        capacity = len(queue)
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.max_priority = max_priority

        for i in range(len(queue)):
            if hasattr(queue[i], "priority"):
                self.add(queue[i].priority, queue[i])
            else:
                queue[i]["priority"] = self.max_priority
                self.add(queue[i]["priority"], queue[i])

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


"""st = SumTree(5)
for i in range(5):
    st.add(i, i)

print(st.tree)
iterations = 1000000
selected_vals = []
for i in range(iterations):
    rand_val = np.random.uniform(0, st.tree[0])
    selected_val = st.get(rand_val)[2]
    selected_vals.append(selected_val)
print(f"Should be ~4: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")

st.update(8, 5)
iterations = 1000000
selected_vals = []
for i in range(iterations):
    rand_val = np.random.uniform(0, st.tree[0])
    selected_val = st.get(rand_val)[2]
    selected_vals.append(selected_val)
print(f"Should be ~5: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")

st.update(8, 0)"""