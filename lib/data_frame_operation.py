import pandas as pd
import setup


def create_data_frame(content_list):
    df = pd.DataFrame(content_list, columns=['id', 'title', 'text'])
    df.to_csv(setup.CSV_FILE_PATH)

    df = pd.read_csv(setup.CSV_FILE_PATH)
    df = df.dropna()
    df.to_csv(setup.CSV_FILE_PATH)

    return df
