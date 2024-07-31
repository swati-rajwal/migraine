import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import nltk
nltk.download('punkt')
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data/B_processed_dataset/combined_dataset.csv')
lengths1 = df['text'].str.split().apply(len)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.histplot(lengths1, bins=50, kde=True, color="skyblue", alpha=0.6, edgecolor='black')
mean_length1 = lengths1.mean()
plt.axvline(mean_length1, color='red', linestyle='dashed', linewidth=2, label=f'Mean text length: {mean_length1:.2f}')
plt.xticks(np.arange(0, lengths1.max() + 10, 10))  # Adjust the max value accordingly if necessary
plt.xlabel('Length of Text (in words)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figures/tweet_lengths.png', dpi=300)

# 54 keywords in total
medication_keywords=[
'topiramate', 'topamax', 'propranolol', 'inderal', 'atenolol', 'tenormin', 'metoprolol', 'toprol', 
'amitriptyline', 'elavil', 'nortriptyline', 'pamelor', 'onabotulinumtoxina', 'botox', 'erenumab', 
'aimovig', 'galcanezumab', 'emgality', 'fremanezumab', 'ajovy', 'eptinezumab', 'vyepti', 'atogepant', 
'qulipta', 'rimegepant', 'nurtec', 'sumatriptan', 'imitrex', 'rizatriptan', 'maxalt', 'eletriptan', 
'relpax', 'naratriptan', 'amerge', 'frovatriptan', 'frova', 'zolmitriptan', 'zomig', 'almotriptan', 
'axert', 'ubrogepant', 'ubrelvy', 'rimegepant', 'nurtec', 'zavegepant', 'zavzpret', 'lasmiditan', 
'reyvow', 'dihydroergotamine', 'dhe', 'migranal', 'trudhesa', 'ergotamine', 'cafergot'
]

med_group_mapping = {
    'topamax': 'antiseizure medications',
    'topiramate': 'antiseizure medications',
    'inderal': 'beta blockers',
    'propranolol': 'beta blockers',
    'tenormin': 'beta blockers',
    'atenolol': 'beta blockers',
    'toprol': 'beta blockers',
    'metoprolol': 'beta blockers',
    'elavil': 'tricyclic antidepressants',
    'amitriptyline': 'tricyclic antidepressants',
    'pamelor': 'tricyclic antidepressants',
    'nortriptyline': 'tricyclic antidepressants',
    'botox': 'onabotulinumtoxina (botox)',
    'onabotulinumtoxina': 'onabotulinumtoxina (botox)',
    'aimovig': 'cgrp monoclonal antibodies',
    'erenumab': 'cgrp monoclonal antibodies',
    'emgality': 'cgrp monoclonal antibodies',
    'galcanezumab': 'cgrp monoclonal antibodies',
    'ajovy': 'cgrp monoclonal antibodies',
    'fremanezumab': 'cgrp monoclonal antibodies',
    'vyepti': 'cgrp monoclonal antibodies',
    'eptinezumab': 'cgrp monoclonal antibodies',
    'qulipta': 'gepants',
    'atogepant': 'gepants',
    'nurtec': 'gepants',
    'rimegepant': 'gepants',
    'imitrex': 'triptans',
    'sumatriptan': 'triptans',
    'maxalt': 'triptans',
    'rizatriptan': 'triptans',
    'relpax': 'triptans',
    'eletriptan': 'triptans',
    'amerge': 'triptans',
    'naratriptan': 'triptans',
    'frova': 'triptans',
    'frovatriptan': 'triptans',
    'zomig': 'triptans',
    'zolmitriptan': 'triptans',
    'axert': 'triptans',
    'almotriptan': 'triptans',
    'ubrelvy': 'gepants',
    'ubrogepant': 'gepants',
    'zavzpret': 'gepants',
    'zavegepant': 'gepants',
    'reyvow': 'ditan',
    'lasmiditan': 'ditan',
    'dhe': 'ergots',
    'dihydroergotamine': 'ergots',
    'migranal': 'ergots',
    'trudhesa': 'ergots',
    'cafergot': 'ergots',
    'ergotamine': 'ergots'
}


# medication_type_mapping = {
#     'antiseizure medications': 'migraine preventive medications',
#     'beta blockers': 'migraine preventive medications',
#     'tricyclic antidepressants': 'migraine preventive medications',
#     'onabotulinumtoxina (botox)': 'migraine preventive medications',
#     'cgrp monoclonal antibodies': 'migraine preventive medications',
#     'gepants': 'migraine acute medications',  # Note that Gepants appears in both Preventive and Acute
#     'triptans': 'migraine acute medications',
#     'ditan': 'migraine acute medications',
#     'ergots': 'migraine acute medications'
# }

medication_type_mapping = {
    'topiramate': 'migraine preventive medication',
    'propranolol': 'migraine preventive medication',
    'atenolol': 'migraine preventive medication',
    'metoprolol': 'migraine preventive medication',
    'amitriptyline': 'migraine preventive medication',
    'nortriptyline': 'migraine preventive medication',
    'onabotulinumtoxina': 'migraine preventive medication',
    'erenumab': 'migraine preventive medication',
    'galcanezumab': 'migraine preventive medication',
    'fremanezumab': 'migraine preventive medication',
    'eptinezumab': 'migraine preventive medication',
    'atogepant': 'migraine preventive medication',
    'rimegepant': 'migraine preventive medication',
    'sumatriptan': 'migraine acute medication',
    'rizatriptan': 'migraine acute medication',
    'eletriptan': 'migraine acute medication',
    'naratriptan': 'migraine acute medication',
    'frovatriptan': 'migraine acute medication',
    'zolmitriptan': 'migraine acute medication',
    'almotriptan': 'migraine acute medication',
    'ubrogepant': 'migraine acute medication',
    'rimegepant': 'migraine acute medication',
    'zavegepant': 'migraine acute medication',
    'lasmiditan': 'migraine acute medication',
    'dihydroergotamine': 'migraine acute medication',
    'ergotamine': 'migraine acute medication'
}

# Dictionary mapping brand names to generic names
brand_to_generic = {
    "topamax": "topiramate",
    "inderal": "propranolol",
    "tenormin": "atenolol",
    "toprol": "metoprolol",
    "elavil": "amitriptyline",
    "pamelor": "nortriptyline",
    "botox": "onabotulinumtoxina",
    "aimovig": "erenumab",
    "emgality": "galcanezumab",
    "ajovy": "fremanezumab",
    "vyepti": "eptinezumab",
    "qulipta": "atogepant",
    "nurtec": "rimegepant",
    "imitrex": "sumatriptan",
    "maxalt": "rizatriptan",
    "relpax": "eletriptan",
    "amerge": "naratriptan",
    "frova": "frovatriptan",
    "zomig": "zolmitriptan",
    "axert": "almotriptan",
    "ubrelvy": "ubrogepant",
    "zavzpret": "zavegepant",
    "reyvow": "lasmiditan",
    "dhe": "dihydroergotamine",
    "migranal": "dihydroergotamine",
    "trudhesa": "dihydroergotamine",
    "cafergot": "ergotamine"
}


def sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

# Function to find medication keywords using regex
def find_keywords(text):
    words = set(word.lower() for word in re.findall(r'\b\w+\b', text))
    found_keywords = {keyword for keyword in medication_keywords if keyword in words}
    return ', '.join(found_keywords)

# Function to map keywords to medication groups
def map_to_med_group(keywords):
    if keywords:
        groups = {med_group_mapping[keyword] for keyword in keywords.split(', ') if keyword in med_group_mapping}
        return ', '.join(groups)
    return ''

def convert_brand_to_generic(keywords):
    # Split keywords by comma and strip spaces
    keywords_list = keywords.split(', ')
    # Replace brand names with generic names
    generic_list = [brand_to_generic.get(keyword, keyword) for keyword in keywords_list]
    # Join the list back into a string
    return ', '.join(generic_list)
    
    
# Function to classify as preventive or acute medication
def classify_medication_type(groups):
    if groups:
        types = {medication_type_mapping[group] for group in groups.split(', ') if group in medication_type_mapping}
        return ', '.join(types)
    return ''

df['text'] = df['text'].str.lower()  # Convert entire text column to lowercase
df['sentiment_score'] = df['text'].apply(sentiment_score)
df['matched_keywords'] = df['text'].apply(find_keywords)
df['generic_keywords'] = df['matched_keywords'].apply(convert_brand_to_generic)
df['med_group'] = df['matched_keywords'].apply(map_to_med_group)
df['prev_acute'] = df['generic_keywords'].apply(classify_medication_type)

df.to_csv('results/migraine_med_sentiments.csv', index=False)
print("migraine_med_sentiments.csv file saved in results folder!")

def count_keywords(keyword_column):
    """Counts occurrences of each keyword in the keyword column of the DataFrame."""
    keyword_counts = {}
    for keywords in keyword_column:
        for keyword in keywords.split(', '):
            if keyword:  # This checks if the keyword is not an empty string
                if keyword in keyword_counts:
                    keyword_counts[keyword] += 1
                else:
                    keyword_counts[keyword] = 1
    return keyword_counts

# Count the keywords using the function
keyword_counts = count_keywords(df['generic_keywords'])
keyword_counts = {key: val for key, val in keyword_counts.items() if val > 0}
print(keyword_counts)
plt.figure(figsize=(12, 8))
plt.barh(list(keyword_counts.keys()), list(keyword_counts.values()), color='blue')
plt.ylabel('Medication Keywords', fontsize=12)  # Adjusted to be the y-axis label
plt.xlabel('Frequency', fontsize=12)  # Adjusted to be the x-axis label
plt.title('Frequency of Generic Migraine Medication in Texts', fontsize=16)
plt.yticks(rotation=0)  # Ensures y-axis labels are horizontal (might not be necessary)
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.grid(True)
plt.savefig('figures/freq_medication_keywords.png')


med_group = count_keywords(df['med_group'])
# print(med_group)
plt.figure(figsize=(12, 8))
plt.barh(list(med_group.keys()), list(med_group.values()), color='blue')
plt.ylabel('Medication Groups', fontsize=12)
plt.xlabel('Frequency', fontsize=12)
plt.title('Frequency of Each Medication Group in Texts', fontsize=16)
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.grid(True)
plt.savefig('figures/freq_medication_groups.png')


group_type = count_keywords(df['prev_acute'])
# print(group_type)
plt.figure(figsize=(12, 8))
plt.bar(group_type.keys(), group_type.values(), color='blue')
plt.xlabel('Medication Medication', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Preventive v/s Acute Migraine Medications', fontsize=16)
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.grid(True)
plt.savefig('figures/freq_medication_prev_acute.png')


# Function to expand the comma-separated entries into individual rows
def expand_rows(df, column_name):
    s = df.apply(lambda x: pd.Series(x[column_name].split(',')), axis=1).stack().reset_index(level=1, drop=True)
    s.name = column_name
    return df.drop(column_name, axis=1).join(s)

# Expanding "matched_keywords", "med_group", and "prev_acute"
df_keywords = expand_rows(df, 'generic_keywords')
df_med_groups = expand_rows(df, 'med_group')
df_prev_acute = expand_rows(df, 'prev_acute')

# Strip extra whitespace
df_keywords['generic_keywords'] = df_keywords['generic_keywords'].str.strip()
df_keywords = df_keywords[df_keywords['generic_keywords'] != '']

df_med_groups['med_group'] = df_med_groups['med_group'].str.strip()
df_med_groups = df_med_groups[df_med_groups['med_group'] != '']

df_prev_acute['prev_acute'] = df_prev_acute['prev_acute'].str.strip()
df_prev_acute = df_prev_acute[df_prev_acute['prev_acute'] != '']

plt.figure(figsize=(18, 14))
sns.boxplot(y='generic_keywords', x='sentiment_score', data=df_keywords, orient='h', palette='coolwarm')
plt.title('Sentiment Scores Distribution Across Migraine Medication Keywords', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=14)
plt.ylabel('Matched Keywords', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
plt.savefig('figures/vader_sentiment_across_keywords.png')
plt.close()

plt.figure(figsize=(21, 14))
sns.boxplot(y='med_group', x='sentiment_score', data=df_med_groups, orient='h', palette='viridis')
plt.title('Sentiment Scores Distribution Across Migraine Medication Groups', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=14)
plt.ylabel('Medication Groups', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
plt.savefig('figures/vader_sentiment_across_groups.png')
plt.close()

plt.figure(figsize=(24, 7))
sns.boxplot(y='prev_acute', x='sentiment_score', data=df_prev_acute, orient='h', palette='magma')
plt.title('Sentiment Scores Distribution Across Preventive/Acute Migraine Medications', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=14)
plt.ylabel('Category', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
plt.savefig('figures/vader_sentiment_across_categories.png')
plt.close()

# KDE Plot
plt.figure(figsize=(15, 11))
sns.kdeplot(data=df_med_groups, x='sentiment_score', hue='med_group', multiple='layer',clip=[-1, 1])
plt.title('Density Plot of Sentiment Scores by Medication Group')
plt.xlabel('Sentiment Score')
plt.ylabel('Density')
plt.savefig('figures/kde_sentiment_scores.png', dpi=300)
plt.close()
