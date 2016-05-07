
import pandas as pd
import os

from scipy.spatial.distance import cosine
from IPython import embed


DATA_PATH = './data/stretch_data.txt'
PROGRESS_PATH = './data/progress_table.csv'
SIMILARITY_PATH = './data/similarity_table.csv'


def generate_progress_table():
    """TODO: map status"""
    df = pd.read_csv(DATA_PATH)
    #df['LASTSTATUSID'] = df['LASTSTATUSID'].replace({})
    progress_table = df.pivot_table(index='ORDERID', columns='CRID', values='LASTSTATUSID').fillna(0)
    return progress_table


def generate_similarity_table():
    progress_table = generate_progress_table()
    df = progress_table.corr()
    df = df.where(df > 0.1, 0)
    return df


def find_recommendations(progress_table, similarity_table, career_id):
    """Return recommend_vector for a career"""

    # Filter orders already applied
    career_vec = progress_table[career_id]
    filter_array = career_vec.where(career_vec == 0, -1)
    filter_array = filter_array.where(filter_array < 0, 1)
    filter_array = filter_array.where(filter_array > 0, 0)

    return progress_table.dot(similarity_table[career_id].values) * filter_array


if __name__ == '__main__':
    if not os.path.exists(PROGRESS_PATH):
        generate_progress_table().to_csv(PROGRESS_PATH)
    if not os.path.exists(SIMILARITY_PATH):
        generate_similarity_table().to_csv(SIMILARITY_PATH)

    progress_table = pd.read_csv(PROGRESS_PATH, index_col='ORDERID')
    similarity_table = pd.read_csv(SIMILARITY_PATH, index_col='CRID')

    # Get recommended order for career '14'
    find_recommendations(progress_table, similarity_table, '14').to_csv('tmp/recommend_vector_14.csv')

