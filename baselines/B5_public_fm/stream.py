"""Resumable per-source shuffled streams; source RNG is separate from FM noise."""
import random
import copy


class PairedStream:
    def __init__(self, sizes, probability, seed):
        if set(sizes) != {"ptbxl", "chapman"} or min(sizes.values()) < 1 or not 0 <= probability <= 1:
            raise ValueError("Invalid public source sizes/probability")
        self.sizes, self.probability = dict(sizes), float(probability)
        self.rng = random.Random(seed)
        self.order = {s: [] for s in sizes}
        self.position = {s: 0 for s in sizes}
        self.visits = {s: 0 for s in sizes}
        self.covered = {s: set() for s in sizes}

    def next(self, batch_size):
        result = []
        for _ in range(batch_size):
            source = "chapman" if self.rng.random() < self.probability else "ptbxl"
            if self.position[source] == len(self.order[source]):
                self.order[source] = list(range(self.sizes[source]))
                self.rng.shuffle(self.order[source])
                self.position[source] = 0
            index = self.order[source][self.position[source]]
            self.position[source] += 1
            self.visits[source] += 1
            self.covered[source].add(index)
            result.append((source, index))
        return result

    def state_dict(self):
        return copy.deepcopy({"sizes": self.sizes, "probability": self.probability, "rng": self.rng.getstate(),
                "order": self.order, "position": self.position, "visits": self.visits,
                "covered": {s: sorted(v) for s, v in self.covered.items()}})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        if state["sizes"] != self.sizes or state["probability"] != self.probability:
            raise ValueError("Cannot resume a different public source mixture")
        self.rng.setstate(state["rng"])
        self.order, self.position, self.visits = state["order"], state["position"], state["visits"]
        self.covered = {s: set(v) for s, v in state["covered"].items()}

    def coverage(self):
        return {s: {"visits": self.visits[s], "unique_records": len(self.covered[s]),
                    "total_records": n, "coverage_fraction": len(self.covered[s]) / n} for s, n in self.sizes.items()}
