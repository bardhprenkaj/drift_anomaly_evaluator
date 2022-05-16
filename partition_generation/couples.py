from partition_generation.abstract.partition import AbstractPartitioner
import numpy as np


class OneAfterTheOtherPartitioner(AbstractPartitioner):

    def __init__(self, array):
        super().__init__(array)

    def partition(self):
        return np.array([[self.array[i-1], self.array[i]] for i in range(len(self.array)-1, 0, -1)])
