
import pandas as pd
from scipy.spatial.distance import cosine


DATA_PATH = './data/stretch_data.txt'


def generate_progress_table():
    df = pd.read_csv(DATA_PATH)
    progress_table = df.pivot_table(index='ORDERID', columns='CRID', values='LASTSTATUSID').fillna(0)
    return progress_table


def generate_similarity_table():
    progress_table = generate_progress_table()
    df = progress_table.corr()
    df = df.where(df > 0.1, 0)
    return df


def generate_cosine_similarity_table():
    """Duplicated method"""
    df = pd.DataFrame(columns=careers, index=careers)
    for c in careers:
        for cc in careers:
            df.ix[c, cc] = get_similarity(progress_table, c, cc)
    return df


def get_similarity(progress_table, sub1, sub2):
    return 1 - cosine(progress_table[sub1], progress_table[sub2])


if __name__ == '__main__':
    generate_progress_table().to_csv('tmp/progress_table.csv')
    generate_similarity_table().to_csv('tmp/similarity_table.csv')

