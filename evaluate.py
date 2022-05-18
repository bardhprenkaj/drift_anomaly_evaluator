from drift_metrics.overlapping_areas.box_intersection import BoxIntersectionMetric
from partition_generation.couples import OneAfterTheOtherPartitioner
import numpy as np

def evaluate(y_true, y_pred, series_len, debug=False):
    prs = []
    recs = []
    fs = []

    if len(y_true) > 1 and len(y_pred) > 1: mid_points = [0] + [(y_true[i+1][0] + y_true[i][1]) // 2 for i in range(len(y_true)-1)] + [series_len-1]
    else: mid_points = [0, series_len - 1]

    if debug: print("mid_points", mid_points)

    if len(y_pred) == 1:
        y_pred_parts = [y_pred for i in range(0, len(mid_points))]
    else:
        y_pred_parts = [[v for v in y_pred if (v <= mid_points[i] and v >= mid_points[i-1])] for i in range(1, len(mid_points))]

    if debug: print("y_pred_part", y_pred_parts)

    for i in range(len(y_pred_parts)):
      if len(y_pred_parts[i]) == 1:
        if i != len(y_pred_parts)-1: y_pred_parts[i].append(y_pred_parts[i+1][0])
        else: y_pred_parts[i].append(mid_points[i+1])

    if debug: print("y_pred_part", y_pred_parts)

    if len(y_pred_parts[-1]) == 0: y_pred_parts[-2][1] = series_len - 1

    i = 0
    for y_pred_p, y_true_p in zip(y_pred_parts, y_true):

        partitioner = OneAfterTheOtherPartitioner(np.array(y_pred_p))
        y_pred_p = partitioner.partition()

        if debug: print(f'y_pred={y_pred_p}')
        if debug: print(f'y_true[i]={y_true_p}')

        oam = BoxIntersectionMetric(np.array([y_true_p]), y_pred_p)

        r, rec_per_dim = oam.recall()
        recs.append(r)
        p, prec_per_dim = oam.precision()
        prs.append(p)
        f = oam.f_score()
        fs.append(f)
        
        i+=1

    p, r, f = np.mean(np.array(prs)), np.mean(np.array(recs)), np.mean(np.array(fs))

    if debug:
      print(f'rec={r}, rec_per_dim={recs}')
      print(f'prec={p}, prec_per_dim={prs}')
      print(f'f1_score={f}')
    
    return p, r, f