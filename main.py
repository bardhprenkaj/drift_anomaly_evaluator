from drift_anomaly_evaluator.evaluate import evaluate

if __name__ == '__main__':
    # partitioner = OneAfterTheOtherPartitioner(np.array([0, 260, 265, 270, 290, 300]))
    # y_pred = partitioner.partition()
    
    # print(f'y_pred={y_pred}')
    
    # oam = BoxIntersectionMetric(np.array([[90, 260], [270, 300]]), y_pred)

    # rec, rec_per_dim = oam.recall()
    # prec, prec_per_dim = oam.precision()
    # f = oam.f_score()
    
    # print(f'rec={rec}, rec_per_dim={rec_per_dim}')
    # print(f'prec={prec}, prec_per_dim={prec_per_dim}')
    # print(f'f1_score={f}')

    len_signal = 1500
    y_true = [[90, 260], [270, 300]]
    y_pred = [0, 260, 265, 270, 290, 300]

    p,r,f = evaluate(y_true, y_pred, len_signal, debug=True)