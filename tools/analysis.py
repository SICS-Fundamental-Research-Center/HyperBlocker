# -*- coding: utf-8 -*-
import pandas as pd
import Levenshtein as lv
import matplotlib.pyplot as plt
import sys
import numpy as np


# title
def Jaro_Winkler(str1, str2):
    if str1 == '' and str2 == '':
        return 1.0
    else:
        return lv.jaro_winkler(str1, str2)


# genre
def jaccard_sim(str1, str2):
    if str1 == '' and str2 == '':
        return 1.0
    a = set(str1.split(','))
    b = set(str2.split(','))
    c = a.intersection(b)
    return len(c) / (len(a) + len(b) - len(c))


if __name__ == '__main__':
    map_pth = sys.argv[1]
    l_pth = sys.argv[2]
    r_pth = sys.argv[3]
    l_csv = pd.read_csv(l_pth, index_col=False, encoding="utf-8")
    r_csv = pd.read_csv(r_pth, index_col=False, encoding='utf-8')
    map_csv = pd.read_csv(map_pth, index_col=False, encoding='utf-8')
    #pd.set_option('display.max_columns', None)

    print(map_csv)
    print(l_csv)
    print(r_csv)
    sim_vec = []
    all_graph_sim_vec = []
    sim_graph = []
    row_num = map_csv.shape[0]
    col_num = l_csv.shape[1] - 1
    count = 0
    sim_graph_p = pd.DataFrame(
        columns=['title', 'release', 'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'year'])
    # sim_graph_p = pd.DataFrame(columns=['title', 'discription', 'manufacturer', 'price'])

    for index, row in map_csv.iterrows():
        l_slice = l_csv.loc[l_csv['id'] == (row[0])]
        r_slice = r_csv.loc[r_csv['id'] == (row[1])]
        sim_tuple = []
        #del l_slice['id']
        if (l_slice.empty or r_slice.empty):
            continue
        for i in l_slice:
            l = l_slice[i].to_numpy(dtype=object)[0]
            r = r_slice[i].to_numpy(dtype=object)[0]

            sim_ = Jaro_Winkler(str(l), str(r))
            # sim_ = jaccard_sim(str(l), str(r))
            # sim_ = lv.distance(str(l), str(r))
            sim_tuple.append(sim_)
        new = pd.DataFrame({'title': sim_tuple[0],
                            'release': sim_tuple[1],
                            'artist_name': sim_tuple[2],
                            'duration': sim_tuple[3],
                            'artist_familiarity': sim_tuple[4],
                            'artist_hotttnesss': sim_tuple[5],
                            'year': sim_tuple[4]},
                           index=[1])
        #	if(sim_tuple[2]<0.4):
        #		print(l)
        #		print(r)

        #	new = pd.DataFrame({ 'title': sim_tuple[0],
        #						'discription': sim_tuple[1],
        #						'manufacturer': sim_tuple[2],
        #						'price': sim_tuple[3]},
        #						index=[1])

        sim_graph_p = sim_graph_p.append(new, ignore_index=True)
        count = count + 1
        if count == 1000:
            break
    x_ = [i for i in range(len(sim_graph_p))]
    tau_ = 0.90
    for i in sim_graph_p:
        y_ = sim_graph_p[i].to_numpy()
        y_ = sorted(y_)
        print("%s: %.1f percentage tuples behind threshold: " % (i, tau_ * 100), y_[int((1 - tau_) * len(y_))])
        plt.scatter(x_, y_, marker='.', label=i)

    plt.legend(loc='lower right', ncol=2, fontsize=10)
    plt.show()


def compute_ndcg():
    from sklearn.metrics import ndcg_score
    true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    scores = np.asarray([[.1, .2, .3, 4, 70]])
    ndcg = ndcg_score(true_relevance, scores)
    print(ndcg)

compute_ndcg()