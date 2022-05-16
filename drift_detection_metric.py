import numpy as np
from shapely.geometry import box


class PartitionBipartiteGraph:

    def __init__(self, partition_a, partition_b):
        self.partition_a = partition_a
        self.partition_b = partition_b

        self.nodes = len(self.partition_a) + len(self.partition_b)

    def build_adjacency_matrix(self, rule):
        self.adj_matrix = np.zeros((self.nodes, self.nodes))

        for i in range(len(self.partition_a)):
            p_a = sorted(self.partition_a[i].tolist())

            rect_p_a = box(p_a[0], 0, p_a[1], 1)

            for j in range(len(self.partition_b)):

                p_b = (self.partition_b[j].tolist())

                if rule(p_b):

                    p_b = sorted(p_b)

                    rect_p_b = box(p_b[0], 0, p_b[1], 1)

                    intersect = rect_p_a.intersection(rect_p_b)

                    self.adj_matrix[i][j] = intersect.area / rect_p_a.area

    def maximum_matching(self):
        return np.max(self.adj_matrix)


class DriftMetrics:

    def __init__(self, y_true: np.array, y_pred: np.array):
        self.y_true = y_true
        self.y_pred = y_pred

    def __partition(self, collection: list) -> list:
        if len(collection) == 1:
            yield [collection]
            return

        first = collection[0]
        for smaller in self.__partition(collection[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n+1:]
            yield [[first]] + smaller

    def is_pair_partition(self, partition):
        return len(partition) == 2

    def is_partition_within_bounds(self, partition):
        if not self.is_pair_partition(partition):
            return False
        elif not self.__check_ranges(partition):
            return False
        return True

    def __check_ranges(self, partition):
        checkers = []
        for y_t in self.y_true:
            checkers.append(partition[0] >= y_t[0] or partition[1] <= y_t[1])
        return any(checkers)

    def precision(self) -> float:
        maximum_matchings_per_partition = list()
        for p in self.__partition(self.y_pred.tolist()):
            pbg = PartitionBipartiteGraph(
                self.y_true, np.array([np.array(x) for x in p]))
            pbg.build_adjacency_matrix(rule=self.is_pair_partition)
            maximum_matchings_per_partition.append(pbg.maximum_matching())
        try:
            return sum(maximum_matchings_per_partition) / len(maximum_matchings_per_partition) + 0.5
        except:
            return 0

    def recall(self):
        maximum_matchings_per_partition = list()
        for p in self.__partition(self.y_pred.tolist()):
            pbg = PartitionBipartiteGraph(
                self.y_true, np.array([np.array(x) for x in p]))
            pbg.build_adjacency_matrix(rule=self.is_partition_within_bounds)
            maximum_matchings_per_partition.append(pbg.maximum_matching())
        try:
            return sum(maximum_matchings_per_partition) / len(maximum_matchings_per_partition)
        except:
            return 0

    def f_score(self, beta=1):
        r = self.recall()
        p = self.precision()

        beta_squared = beta ** 2
        return (1 + beta_squared) * (p*r) / ((beta_squared * p) + r)


if __name__ == '__main__':

    metrics = DriftMetrics(
        np.array([[366, 755], [200, 254]]), np.array([200, 500, 750, 832]))

    print(metrics.precision())

    print(metrics.recall())
