import pandas as pd
import sys
import cmath
import numpy as np
from datasketch import MinHash

def get_lsh_hash_value(text):
    m = MinHash(num_perm=3)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def read_file(input_path1):
    """
    Reads the file and returns a dataframe
    """
    df1 = pd.read_csv(input_path1)

    cdf1 = np.zeros(32)

    col_name = "venue"


    minhash = get_lsh_hash_value(col_name)
    hash_value = minhash.hashvalues
    print(hash_value)


    minhash = get_lsh_hash_value(col_name)
    hash_value = minhash.hashvalues
    print(hash_value)



    for index, row in df1.iterrows():
        minhash = get_lsh_hash_value(str(row[col_name]))
        hash_value = minhash.hashvalues
        for i in hash_value:
            bucket_id = int(i) % 32
            cdf1[bucket_id] = cdf1[bucket_id] + 1


    avg = df1.shape[0] / 32 * 3
    avg_cdf = np.ones(32) * avg

    print(avg_cdf)


    print(sum(np.sqrt ((cdf1 - avg_cdf) * (cdf1 - avg_cdf))))
    return


if __name__ == '__main__':
    read_file(input_path1=sys.argv[1])
