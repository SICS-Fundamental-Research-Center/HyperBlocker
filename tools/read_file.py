import pandas as pd
import sys
def read_file(input_path, output_path):
    """
    Reads the file and returns a dataframe
    """
    df = pd.read_csv(input_path)
    print(df)
    print(df.head())
    df.sort_values(by=['id'], inplace=True)

    df.to_csv(output_path, index=False,sep='|')
    return df

if __name__ == '__main__':
    read_file(input_path=sys.argv[1], output_path=sys.argv[2])
