import pandas as pd
import numpy as np
import sys


def output2match(data_path_l, data_path_r, result_path, output_path):
    print(data_path_l)
    print(data_path_r)
    # csv_l = pd.read_csv(data_path_l, index_col=False, encoding="utf-8")
    csv_l = pd.read_csv(data_path_l)
    # csv_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')
    csv_r = pd.read_csv(data_path_r)
    result = pd.read_csv(result_path, header=None)

    print(csv_l)
    print(csv_r)
    print(result)

    for index, row in result.iterrows():
        print(row[0], row[1])

        #csv_l.loc[csv_l['id'] == row[0], 'label'] = row[1]

    return


def fun(result_path, perfect_match_path):
    result = pd.read_csv(result_path, index_col=False, encoding="utf-8", header=None)
    perfect_match = pd.read_csv(perfect_match_path, index_col=False, encoding='utf-8')

    print(result)
    print(perfect_match)

    count = 0;
    for index, row in result.iterrows():
        perfect_match_to_be_check = perfect_match.loc[perfect_match['idDBLP'] == row[0]]
        final = perfect_match_to_be_check.loc[perfect_match_to_be_check['idACM'] == row[1]]
        if(not final.empty):
            count = count + 1

    print(count)


if __name__ == '__main__':
    #output2match(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    fun(sys.argv[1], sys.argv[2])
    # fun(sys.argv[0], sys.argv[1], sys.argv[2])
