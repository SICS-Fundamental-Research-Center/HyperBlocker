import pandas as pd
import numpy as np
import sys
from bitmap import BitMap


def fun(result_path, perfect_match_path):
    result = pd.read_csv(result_path, index_col=False, encoding="utf-8", header=None)
    perfect_match = pd.read_csv(perfect_match_path, index_col=False, encoding='utf-8', sep=',', header=None)

    print(result)
    # result = result.drop_duplicates(subset=[0, 1], keep='first')
    result = result.drop_duplicates(keep='first')
    perfect_match = perfect_match.drop_duplicates(keep='first')
    print(perfect_match)
    # result.to_csv(result_path + 'no_duplicate.csv', index=False, header=None, encoding='utf-8')

    count = 0;
    mismatch = pd.DataFrame(columns=['id1', 'id2'])
    for index, row in result.iterrows():
        perfect_match_to_be_check = perfect_match.loc[perfect_match[0] == row[0]]
        final = perfect_match_to_be_check.loc[perfect_match_to_be_check[1] == row[1]]

        perfect_match_to_be_check_reverse = perfect_match.loc[perfect_match[1] == row[0]]
        final_reverse = perfect_match_to_be_check_reverse.loc[perfect_match_to_be_check_reverse[0] == row[1]]
        if ((not final.empty) or (not final_reverse.empty)):
            count = count + 1
        else:
            mismatch = mismatch.append({'id1': row[0], 'id2': row[1]}, ignore_index=True)

    print("mismatch")
    print(mismatch)

    TP = count
    FP = len(result) - count

    print("TP", count, " FP", FP)

    precision = TP / len(result)
    recall = TP / len(perfect_match)
    f1 = 2 * precision * recall / (precision + recall)

    print("prec: ", precision, "recall: ", recall, "f1: ", f1)

    precision = 0.8004
    recall = 0.8033
    f1 = 2 * precision * recall / (precision + recall)
    print("self prec: ", precision, "recall: ", recall, "f1: ", f1)


def verify(result_path, data_path_l, data_path_r, output_path):
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8', sep=',')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8', sep=',')

    result = pd.read_csv(result_path, index_col=False, encoding="utf-8", header=None)
    result = result.drop_duplicates(keep='first')
    print(result)

    bm = BitMap(16777216)

    perfect_match = pd.DataFrame(columns=['id1', 'id2'])
    missed_match = pd.DataFrame(columns=['id1', 'id2'])

    for index, row in data_l.iterrows():
        bm.set(row[0])

    data_count = 0
    for index, row in data_r.iterrows():
        if (bm.test(row[0])):
            data_count = data_count + 1
            # perfect_match = perfect_match.append({'id1': row[0], 'id2': row[0]}, ignore_index=True)

    TP = 0
    for index, row in result.iterrows():
        if (row[0] == row[1]):
            TP = TP + 1

    for index, row in perfect_match.iterrows():
        result_to_be_check = result.loc[result[0] == row[0]]
        final = result_to_be_check.loc[result_to_be_check[1] == row[1]]
        if (len(final) == 0):
            missed_match = missed_match.append({'id1': row[0], 'id2': row[1]}, ignore_index=True)
            break

    # print(perfect_match)
    print(TP)
    print(data_count)

    print("missed")
    print(missed_match)
    missed_match.to_csv(output_path, index=False, encoding="utf-8")

    precision = TP / len(result)
    recall = TP / data_count
    f1 = 2 * precision * recall / (precision + recall)
    print("self prec: ", precision, "recall: ", recall, "f1: ", f1)


def GenerateMatches(data_path_l, data_path_r, output_path):
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8', sep=',')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8', sep=',')

    print(data_l)
    print(data_r)
    bm = BitMap(16777216)
    perfect_match = pd.DataFrame(columns=['ltable_id', 'rtable_id'])
    for index, row in data_l.iterrows():
        bm.set(row[0])

    for index, row in data_r.iterrows():
        if (bm.test(row[0])):
            perfect_match = perfect_match.append({'ltable_id': row[0], 'rtable_id': row[0]}, ignore_index=True)

    print(perfect_match)
    perfect_match.to_csv(output_path, index=False, encoding="utf-8")
    return


if __name__ == '__main__':
    fun(sys.argv[1], sys.argv[2])
    # verify(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # GenerateMatches(sys.argv[1], sys.argv[2], sys.argv[3])
