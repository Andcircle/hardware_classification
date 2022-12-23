import pandas as pd

def load_label_csv(path):
    des, labels = [], []
    dataframe = pd.read_csv(path)
    datas=dataframe.to_dict(orient='records')
    for row in datas:
        des.append(row['Description'])
        labels.append(row['Label'])
    return des, labels
