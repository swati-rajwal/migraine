import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df1 = pd.read_excel('data/B_processed_dataset/Copy of sample_0811_part1_EJ_new_YC_022024-1_CC.xlsx')
df2 = pd.read_excel('data/B_processed_dataset/Copy of sample_0811_part2_Trivedi_YC_032524-2_final (1)_CC.xlsx')
df3 = pd.read_excel('data/B_processed_dataset/Copy of setB_full_new_YC_040424-2_finalCC.xlsx')
print(f"ORIGINAL DATASET SHAPE:\ndf1: {df1.shape}\ndf2: {df2.shape}\ndf3: {df3.shape}")

df3.rename(columns={'therapy':'keyword','sentiment':'label','tweet_text':'text'},inplace=True)

for index, row in df2.iterrows():
    if not pd.isnull(row['FINAL sentiment']):
        df2.at[index, 'label'] = row['FINAL sentiment']
    if not pd.isnull(row['efficacy']):
        df2.at[index, 'Efficacy'] = row['efficacy']
    if not pd.isnull(row['tolerability']):
        df2.at[index, 'Tolerability/Side effects'] = row['tolerability']
    if not pd.isnull(row['accessibility']):
        df2.at[index, 'Accessability (insurance denial, \ncomplain have to travel to get treatment)'] = row['accessibility']
    if not pd.isnull(row['others']):
        df2.at[index, 'Others'] = row['others']

for index, row in df3.iterrows():
    if not pd.isnull(row['FINAL sentiment']):
        df3.at[index, 'label'] = row['FINAL sentiment']
    if not pd.isnull(row['efficacy.1']):
        df3.at[index, 'efficacy'] = row['efficacy.1']
    if not pd.isnull(row['tolerability.1']):
        df3.at[index, 'tolerability'] = row['tolerability.1']
    if not pd.isnull(row['accessibility.1']):
        df3.at[index, 'accessibility'] = row['accessibility.1']
    if not pd.isnull(row['others.1']):
        df3.at[index, 'others'] = row['others.1']

df1.drop(['Unnamed: 4','Unnamed: 11'], axis=1, inplace=True)
df2.drop(['Notes','2nd reviewer disagreement', 'FINAL sentiment', 'efficacy',
       'tolerability', 'accessibility', 'others'],axis=1,inplace=True)
df3.drop(['notes (optional)','2nd reviewer disagreement', 'FINAL sentiment', 'efficacy.1',
       'tolerability.1', 'accessibility.1', 'others.1'],axis=1,inplace=True)

df1.columns = map(str.lower, df1.columns)
df2.columns = map(str.lower, df2.columns)
df3.columns = map(str.lower, df3.columns)

df1.rename(columns={'tolerability/side effects':'tolerability', 'accessability (insurance denial, \ncomplain have to travel to get treatment)':'accessibility'},inplace=True)
df2.rename(columns={'tolerability/side effects':'tolerability', 'accessability (insurance denial, \ncomplain have to travel to get treatment)':'accessibility'},inplace=True)


df3['label'] = df3['label'].replace({'Negative': -1, 'Neutral': 0, 'Positive': 1})
df3['efficacy'] = df3['efficacy'].replace({'Negative': -1, 'Neutral': 0, 'Positive': 1})
df3['tolerability'] = df3['tolerability'].replace({'Negative': -1, 'Neutral': 0, 'Positive': 1})
df3['accessibility'] = df3['accessibility'].replace({'Negative': -1, 'Neutral': 0, 'Positive': 1})
df3['others'] = df3['others'].replace({'Negative': -1, 'Neutral': 0, 'Positive': 1})

print(f"################ FIRST EXCEL SHEET DATA ################\ndf1 columns: {df1.columns},\ndf1 shape: {df1.shape}\n")
print(f"################ SECOND EXCEL SHEET DATA ################\ndf2 columns: {df2.columns},\ndf2 shape: {df2.shape}\n")
print(f"################ THIRD EXCEL SHEET DATA ################\ndf3 columns: {df3.columns},\ndf3 shape: {df3.shape}\n")

# combined_df = pd.concat([df1, df2, df3]).drop_duplicates(subset=['tweet_id', 'keyword'])
combined_df = pd.concat([df1, df2, df3])
combined_df.reset_index(drop=True, inplace=True)
combined_df['new_id'] = combined_df['new_id'].str.replace(r'\s+', '', regex=True)
combined_df['sort_key'] = combined_df['filename'] == 'Copy of setB_full_new_YC_040424-2_finalCC'
combined_df.sort_values('sort_key', ascending=False, inplace=True)
combined_df = combined_df.drop_duplicates(subset='new_id', keep='first')
combined_df.drop(columns='sort_key', inplace=True)
combined_df.reset_index(drop=True, inplace=True)
# Remove white space before or after a keyword
'''
' amerge' & 'amerge' should be same
'sumatriptan', 'sumatriptan ' should be same
'''
combined_df['keyword'] = combined_df['keyword'].str.strip()

print(f"Shape of combined dataframe: {combined_df.shape}")

combined_df.to_csv('data/B_processed_dataset/combined_dataset.csv',index=False)

print(combined_df.columns)

print(combined_df.shape)

print("Number of unique medication keywords:", len(combined_df['keyword'].unique()))
print("Unique medication/therapy keywords:", sorted(combined_df['keyword'].unique()))

def plot_graph(colname):
    keyword_counts = combined_df[colname].value_counts()
    plt.figure(figsize=(10, 6))
    keyword_counts.plot(kind='barh')
    plt.title(f'Frequency of values in column: {colname}')
    plt.xlabel('Frequency')
    plt.ylabel(colname)
    plt.grid()
    plt.show()
    plt.savefig(f'figures/{colname}.png')

plot_graph('keyword')
plot_graph('label')

for colname in ['label', 'efficacy', 'tolerability','accessibility', 'others']:
    print(combined_df[colname].value_counts(dropna=False),"\n")

# train_df, temp_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['label'])
# dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

# train_df.reset_index(drop=True, inplace=True)
# dev_df.reset_index(drop=True, inplace=True)
# test_df.reset_index(drop=True, inplace=True)

# train_df.to_csv('results/train.csv',index=False)
# dev_df.to_csv('results/dev.csv',index=False)
# test_df.to_csv('results/test.csv',index=False)
