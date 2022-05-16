from drift_metrics.overlapping_areas.box_intersection import BoxIntersectionMetric
import numpy as np

if __name__ == '__main__':
    oam = BoxIntersectionMetric(np.array([[90, 260], [270, 300]]), np.array([
                                [0, 260], [265, 270], [290, 300]]))
    
    print(oam.true_boxes)
    print(oam.pred_boxes)
    
    
    rec, rec_per_dim = oam.recall()
    prec, prec_per_dim = oam.precision()
    f = oam.f_score()
    
    print(f'rec={rec}, rec_per_dim={rec_per_dim}')
    print(f'prec={prec}, prec_per_dim={prec_per_dim}')
    print(f)
