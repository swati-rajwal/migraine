import pandas as pd

def load_data(
    dat_file,
    is_train=False,
    sep=',',
    text_name='text',
    label_name='label',
    classes = None,     # text label list
    reverse = False, # convert numeric labels to text labels
    load_class = True, 
    ):

    #df = pd.read_csv(dat_file, usecols=[text_name, label_name], sep=sep)
    df = pd.read_csv(dat_file, sep=sep)

    # for trainig data, shuffle the data
    if is_train:
        df = df.sample(frac=1).reset_index(drop=True)

    # simpletransformers needs the label column named 'labels'
    # df = df.dropna()
    df = df.rename(columns={label_name:'labels'})
    #print(df.head())

    # for specific task
    if load_class and classes != None:
        if reverse:
            df['labels'] = df['labels'].apply(lambda i:classes[i])
        else:
            df['labels'] = df['labels'].apply(classes.index)

    print(df.head())
    print('size:', len(df))
    return df


def get_classes(data_dir):
    df = pd.read_csv(f'{data_dir}/train.csv')
    labels = df['label'].values.tolist()
    classes = list(set(labels))
    classes.sort()
    return classes
