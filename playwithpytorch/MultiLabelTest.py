import pandas as pd


def explore_dataset():
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    df = pd.read_csv("{0}\\test.csv".format(path_to_data))
    #print(df.head())
    #print(df['labels'])
    #making distinct list of categories
    cat_list = [labels for labels in df['labels']]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()
    print(label_list)

def generate_encoder_mapping(path_to_data, label_column_name):
    df = pd.read_csv("{0}\\test.csv".format(path_to_data))
    cat_list = [labels for labels in df[label_column_name]]
    label_set = set()
    for labels in cat_list:
        for label in labels.split(' '):
            label_set.add(label)

    label_list = list(label_set)
    label_list.sort()
    return {label_list[idx]: idx for idx in range(len(label_list))}

if __name__ == "__main__":
    path_to_data = r"C:\Users\HFD347\.fastai\data\pascal_2007"
    t = generate_encoder_mapping(path_to_data, "labels")
    print(t)