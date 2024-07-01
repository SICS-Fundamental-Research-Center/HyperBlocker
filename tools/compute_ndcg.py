from sklearn.metrics import ndcg_score
import numpy as np

def compute_ndcg():
    true_relevance = np.asarray([[2,5,3, 4,1,6]])
    scores = np.asarray([[5,2,3,4,1,6]])
    ndcg = ndcg_score(true_relevance, scores)
    print(ndcg)

compute_ndcg()
