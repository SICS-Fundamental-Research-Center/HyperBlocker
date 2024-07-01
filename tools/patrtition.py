import pandas as pd
import sys
import time

def partition_bak(input_path, output_path):
    """
    Reads the file and returns a dataframe
    """
    df = pd.read_csv(input_path)
    print(df)
    print(df.head())
    df.sort_values(by=['id'], inplace=True)


    dt = dict
    #for index, row in df.iterrows():
    #    print(row['year'])

    df.to_csv(output_path, index=False,sep=',')
    return df

def partition(input_path, output_path, n):
    """
    Reads the file and returns a dataframe
    """
    df = pd.read_csv(input_path)
    print(df)
    print(df.head())
    df.sort_values(by=['id'], inplace=True)


    partition = {
        0: pd.DataFrame(),
        1: pd.DataFrame(),
        2: pd.DataFrame(),
        3: pd.DataFrame(),
    }
    dt = dict

    time_start  = time.time()
    for index, row in df.iterrows():
        bucket_id = int(row['year']) % int(n)
        partition[bucket_id] = partition[bucket_id].append(row)
    time_end  = time.time()
    print(time_end - time_start, "s")


    for key in partition.keys():
        partition[int(key)].to_csv(output_path + str(key), index=False, sep=',')

    time_end2  = time.time()
    print(time_end2 - time_start, "s")

    return df

if __name__ == '__main__':
    partition(input_path=sys.argv[1], output_path=sys.argv[2], n = sys.argv[3])
