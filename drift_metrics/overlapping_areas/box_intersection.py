from drift_metrics.abstract.base_metric import AbstractMetric
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
import numpy as np


class BoxIntersectionMetric(AbstractMetric):

    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

        self.true_boxes = BoxIntersectionMetric.boxify(self.y_true)
        self.pred_boxes = BoxIntersectionMetric.boxify(self.y_pred)

    @staticmethod
    def boxify(array):
        boxified_array = np.empty(shape=(array.shape[0],)).astype(Polygon)

        for i in range(array.shape[0]):
            boxified_array[i] = BoxIntersectionMetric.box_area(array[i])

        return boxified_array

    @staticmethod
    def box_area(coords):
        a, b = coords
        return box(a, 0, b, 1)

    def calc_metric(self, recall=True):
        metric = np.empty(shape=(len(self.pred_boxes), len(self.true_boxes)))

        def get_area(pred_box, true_box, recall=True):
            return pred_box.intersection(true_box).area / (true_box.area if recall else pred_box.area)

        for i_pred in range(len(self.pred_boxes)):
            for j_true in range(len(self.true_boxes)):
                metric[i_pred][j_true] = get_area(
                    self.pred_boxes[i_pred],
                    self.true_boxes[j_true],
                    recall=recall)

        max_metric_pred = np.max(metric, axis=1)
        return np.mean(max_metric_pred), max_metric_pred

    def precision(self):
        return self.calc_metric(recall=False)

    def recall(self):
        return self.calc_metric()
