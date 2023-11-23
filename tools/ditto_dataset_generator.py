import pandas as pd
import numpy as np
import sys
import random


def HyperBlockerOutput2DittoDataset(perfect_match_path, data_path_l, data_path_r, output_path):
    perfect_match_path = pd.read_csv(perfect_match_path, index_col=False, encoding="utf-8", header=None)
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')

    print(perfect_match_path)
    print(data_l)
    print(data_r)

    f = open(output_path, 'w')
    col_name = ['title', 'authors', 'venue', 'year']
    for index, row in perfect_match_path.iterrows():
        match_data_l = data_l.loc[data_l['id'] == row[0]]
        match_data_r = data_r.loc[data_r['id'] == row[1]]

        line = "COL " + col_name[0] + " VAL " + str(match_data_l.iloc[0][col_name[0]]) + " COL " + col_name[
            1] + " VAL " + str(match_data_l.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + \
               str(match_data_l.iloc[0][col_name[2]]) + col_name[3] + " VAL " + str(
            match_data_l.iloc[0][col_name[3]]) + " \tCOL " + \
               col_name[0] + " VAL " + match_data_r.iloc[0][col_name[0]] + col_name[1] + " VAL " + str(
            match_data_r.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + str(
            match_data_r.iloc[0][col_name[2]]) + " COL " + col_name[3] + " VAL " + str(
            match_data_r.iloc[0][col_name[3]]) + " \t1"

        f.write(line + '\n')

    f.close()
    return


def HyperBlockerOutput2DittoDatasetSongs(predicted_math_path, perfect_match_path, data_path_l, data_path_r,
                                         output_path):
    perfect_match = pd.read_csv(perfect_match_path, index_col=False, encoding="utf-8", header=None)
    predicted_match = pd.read_csv(predicted_math_path, index_col=False, encoding="utf-8", header=None)
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')

    perfect_match = perfect_match.drop_duplicates(keep='first')
    print(perfect_match)
    print(data_l)
    print(data_r)

    f = open(output_path, 'w')
    col_name = ['title', 'release', 'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'year']
    TP = 0
    for index, row in predicted_match.iterrows():
        match_data_l = data_l.loc[data_l['id'] == row[0]]
        match_data_r = data_r.loc[data_r['id'] == row[1]]

        label = 1

        perfect_match_to_be_check = perfect_match.loc[perfect_match[0] == row[0]]
        final = perfect_match_to_be_check.loc[perfect_match_to_be_check[1] == row[1]]
        perfect_match_to_be_check_reverse = perfect_match.loc[perfect_match[1] == row[0]]
        final_reverse = perfect_match_to_be_check_reverse.loc[perfect_match_to_be_check_reverse[0] == row[1]]
        if ((not final.empty) or (not final_reverse.empty)):
            TP = TP + 1
            label = 1
        else:
            label = 0

        line = "COL " + col_name[0] + " VAL " + str(match_data_l.iloc[0][col_name[0]]) + " COL " + col_name[
            1] + " VAL " + str(match_data_l.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + \
               str(match_data_l.iloc[0][col_name[2]]) + col_name[3] + " VAL " + str(
            match_data_l.iloc[0][col_name[3]]) + col_name[4] + " VAL " + str(
            match_data_l.iloc[0][col_name[4]]) + col_name[5] + " VAL " + str(
            match_data_l.iloc[0][col_name[5]]) + col_name[6] + " VAL " + str(
            match_data_l.iloc[0][col_name[6]]) + " \tCOL " + col_name[0] + " VAL " + str(match_data_r.iloc[0][
                                                                                             col_name[0]]) + col_name[
                   1] + " VAL " + str(match_data_r.iloc[0][col_name[1]]) + " COL " + col_name[
                   2] + " VAL " + str(match_data_r.iloc[0][col_name[2]]) + " COL " + col_name[3] + " VAL " + str(
            match_data_r.iloc[0][col_name[3]]) + " COL " + col_name[4] + " VAL " + str(
            match_data_r.iloc[0][col_name[4]]) + " COL " + col_name[5] + " VAL " + str(
            match_data_r.iloc[0][col_name[5]]) + " COL " + col_name[6] + " VAL " + str(
            match_data_r.iloc[0][col_name[6]]) + " \t" + str(label)
        f.write(line + '\n')

    count = 0

    f.close()
    precision = TP / len(predicted_match)
    recall = TP / len(perfect_match)
    f1 = 2 * precision * recall / (precision + recall)

    print("prec: ", precision, "recall: ", recall, "f1: ", f1)
    return


def HyperBlockerOutput2DittoDatasetNCV(predicted_math_path, data_path_l, data_path_r,
                                       output_path):
    predicted_match = pd.read_csv(predicted_math_path, index_col=False, encoding="utf-8", header=None)
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')

    print(data_l['recid'])
    print(data_r)

    f = open(output_path, 'w')
    col_name = ['givenname', 'surname', 'suburb', 'postcode']
    for index, row in predicted_match.iterrows():
        match_data_l = data_l.loc[data_l['recid'] == int(row[0])]
        match_data_r = data_r.loc[data_r['recid'] == int(row[1])]
        label = 1
        if (row[0] == row[1]):
            label = 1
        else:
            label = 0

        line = "COL " + col_name[0] + " VAL " + str(match_data_l.iloc[0][col_name[0]]) + " COL " \
               + col_name[1] + " VAL " + str(match_data_l.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + str(
            match_data_l.iloc[0][col_name[2]]) + col_name[3] + " VAL " + str(
            match_data_l.iloc[0][col_name[3]]) + " \tCOL " + col_name[0] + " VAL " + str(
            match_data_r.iloc[0][col_name[0]]) + col_name[1] + " VAL " + str(
            match_data_r.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + str(
            match_data_r.iloc[0][col_name[2]]) + " COL " + col_name[3] + " VAL " + str(
            match_data_r.iloc[0][col_name[3]]) + " \t" + str(label)
        f.write(line + '\n')

        count = 0

    f.close()
    return


def RandomJoinDBLPACM(perfect_match_path, data_path_l, data_path_r, output_path):
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')
    perfect_match = pd.read_csv(perfect_match_path, index_col=False, encoding="utf-8", header=None)
    f = open(output_path, 'w')

    print(perfect_match_path)
    print(data_l)
    print(data_r)

    joined_data = pd.merge(data_l, data_r, "inner", left_on=['year'], right_on=['year'])
    print(joined_data)

    col_name = ['title', 'authors', 'venue', 'postcode']
    count = 0
    for index, row in joined_data.iterrows():
        print(row)
        line = "COL " + col_name[0] + " VAL " + str(row['title_x']) + " COL " + col_name[
            1] + " VAL " + str(row['authors_x']) + " COL " + col_name[2] + " VAL " + \
               str(row['venue_x']) + col_name[3] + " VAL " + str(row['year']) + " \tCOL " + \
               col_name[0] + " VAL " + str(row['title_y']) + col_name[1] + " VAL " + str(row['authors_y']) + " COL " + \
               col_name[2] + " VAL " + str(row['venue_y']) + " COL " + col_name[3] + " VAL " + str(row['year']) + " \t1"
        f.write(line + '\n')

    for index, row in perfect_match.iterrows():
        match_data_l = data_l.loc[data_l['id'] == row[0]]
        match_data_r = data_r.loc[data_r['id'] == row[1]]
        line = "COL " + col_name[0] + " VAL " + str(match_data_l.iloc[0][col_name[0]]) + " COL " + col_name[
            1] + " VAL " + str(match_data_l.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + \
               str(match_data_l.iloc[0][col_name[2]]) + col_name[3] + " VAL " + str(
            match_data_l.iloc[0][col_name[3]]) + " \tCOL " + \
               col_name[0] + " VAL " + match_data_r.iloc[0][col_name[0]] + col_name[1] + " VAL " + str(
            match_data_r.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + str(
            match_data_r.iloc[0][col_name[2]]) + " COL " + col_name[3] + " VAL " + str(
            match_data_r.iloc[0][col_name[3]]) + " \t1"
        f.write(line + '\n')

    f.close()

    return


def RandomJoinNCV(data_path_l, data_path_r, output_path):
    data_l = pd.read_csv(data_path_l, index_col=False, encoding='utf-8')
    data_r = pd.read_csv(data_path_r, index_col=False, encoding='utf-8')
    f = open(output_path, 'w')

    print(data_l)
    print(data_r)

    joined_data = pd.merge(data_l, data_r, "inner", left_on=['recid'], right_on=['recid'])
    print(joined_data)

    col_name = ['givenname', 'surname', 'suburb', 'postcode']
    # count = 0
    for i in range(joined_data.shape[0]):
        line = "COL " + col_name[0] + " VAL " + str(data_l.iloc[i]['givenname']) + " COL " + col_name[
            1] + " VAL " + str(data_l.iloc[i]['surname']) + " COL " + col_name[2] + " VAL " + str(
            data_l.iloc[i]['suburb']) + col_name[3] + " VAL " + str(
            data_l.iloc[i]['postcode']) + " \tCOL " + col_name[0] + " VAL " + str(
            data_r.iloc[i]['givenname']) + " COL " + col_name[1] + " VAL " + str(data_r.iloc[i]['surname']) + " COL " + \
               col_name[2] + " VAL " + str(data_r.iloc[i]['suburb']) + col_name[3] + " VAL " + str(
            data_r.iloc[i]['postcode']) + " \t0"
        f.write(line + '\n')
    for index, row in joined_data.iterrows():
        line = "COL " + col_name[0] + " VAL " + str(row['givenname_x']) + " COL " + col_name[
            1] + " VAL " + str(row['surname_x']) + " COL " + col_name[2] + " VAL " + \
               str(row['suburb_x']) + col_name[3] + " VAL " + str(row['postcode_x']) + " \tCOL " + \
               col_name[0] + " VAL " + str(row['givenname_y']) + col_name[1] + " VAL " + str(
            row['surname_y']) + " COL " + \
               col_name[2] + " VAL " + str(row['suburb_y']) + " COL " + col_name[3] + " VAL " + str(
            row['postcode_y']) + " \t1"
        f.write(line + '\n')

    # for index, row in perfect_match.iterrows():
    #    match_data_l = data_l.loc[data_l['id'] == row[0]]
    #    match_data_r = data_r.loc[data_r['id'] == row[1]]
    #    line = "COL " + col_name[0] + " VAL " + str(match_data_l.iloc[0][col_name[0]]) + " COL " + col_name[
    #        1] + " VAL " + str(match_data_l.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + \
    #           str(match_data_l.iloc[0][col_name[2]]) + col_name[3] + " VAL " + str(
    #        match_data_l.iloc[0][col_name[3]]) + " \tCOL " + \
    #           col_name[0] + " VAL " + match_data_r.iloc[0][col_name[0]] + col_name[1] + " VAL " + str(
    #        match_data_r.iloc[0][col_name[1]]) + " COL " + col_name[2] + " VAL " + str(
    #        match_data_r.iloc[0][col_name[2]]) + " COL " + col_name[3] + " VAL " + str(
    #        match_data_r.iloc[0][col_name[3]]) + " \t1"
    #    f.write(line + '\n')

    # f.close()

    return


if __name__ == '__main__':
    HyperBlockerOutput2DittoDatasetNCV(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # HyperBlockerOutput2DittoDataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # HyperBlockerOutput2DittoDatasetSongs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # RandomJoinDBLPACM(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # RandomJoinNCV(sys.argv[1], sys.argv[2], sys.argv[3])
