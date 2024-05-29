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

medication_keywords = [
    'Topiramate', 'Topamax', 'Propranolol', 'Inderal', 'Amitriptyline', 'Elavil',
    'OnabotulinumtoxinA', 'Botox', 'CGRP monoclonal antibodies', 'erenumab', 'Aimovig',
    'galcanezumab', 'Emgality', 'fremanezumab', 'Ajovy', 'eptinezumab', 'Vyepti',
    'Gepants', 'Atogepant', 'Qulipta', 'rimegepant', 'Nurtec', 'Triptans', 'Sumatriptan',
    'Imitrex', 'Rizatriptan', 'Maxalt', 'Eletriptan', 'Relpax', 'Naratriptan', 'Amerge',
    'Frovatriptan', 'Frova', 'Zolmitriptan', 'Zomig', 'Almotriptan', 'Axert', 'ubrogepant',
    'Ubrelvy', 'zavegepant', 'Zavzpret', 'Ditan', 'lasmiditan', 'Reyvow', 'Ergots',
    'Dihydroergotamine', 'DHE', 'Migranal', 'Trudhesa', 'ergotamine', 'Cafergot'
]
med_group_mapping = {
    'Topiramate': 'Topiramate (Topamax)', 'Topamax': 'Topiramate (Topamax)',
    'Propranolol': 'Propranolol (Inderal)', 'Inderal': 'Propranolol (Inderal)',
    'Amitriptyline': 'Tricyclic antidepressants', 'Elavil': 'Tricyclic antidepressants',
    'OnabotulinumtoxinA': 'OnabotulinumtoxinA (Botox)', 'Botox': 'OnabotulinumtoxinA (Botox)',
    'erenumab': 'CGRP monoclonal antibodies', 'Aimovig': 'CGRP monoclonal antibodies',
    'galcanezumab': 'CGRP monoclonal antibodies', 'Emgality': 'CGRP monoclonal antibodies',
    'fremanezumab': 'CGRP monoclonal antibodies', 'Ajovy': 'CGRP monoclonal antibodies',
    'eptinezumab': 'CGRP monoclonal antibodies', 'Vyepti': 'CGRP monoclonal antibodies',
    'Atogepant': 'Gepants', 'Qulipta': 'Gepants', 'rimegepant': 'Gepants', 'Nurtec': 'Gepants',
    'Sumatriptan': 'Triptans', 'Imitrex': 'Triptans',
    'Rizatriptan': 'Triptans', 'Maxalt': 'Triptans',
    'Eletriptan': 'Triptans', 'Relpax': 'Triptans',
    'Naratriptan': 'Triptans', 'Amerge': 'Triptans',
    'Frovatriptan': 'Triptans', 'Frova': 'Triptans',
    'Zolmitriptan': 'Triptans', 'Zomig': 'Triptans',
    'Almotriptan': 'Triptans', 'Axert': 'Triptans',
    'ubrogepant': 'Gepants', 'Ubrelvy': 'Gepants',
    'zavegepant': 'Gepants', 'Zavzpret': 'Gepants',
    'lasmiditan': 'Ditan', 'Reyvow': 'Ditan',
    'Dihydroergotamine': 'Ergots', 'DHE': 'Ergots', 'Migranal': 'Ergots', 'Trudhesa': 'Ergots',
    'ergotamine': 'Ergots', 'Cafergot': 'Ergots'
}
medication_type_mapping = {
    'Topiramate (Topamax)': 'Migraine Preventive Medications',
    'Propranolol (Inderal)': 'Migraine Preventive Medications',
    'Tricyclic antidepressants': 'Migraine Preventive Medications',
    'OnabotulinumtoxinA (Botox)': 'Migraine Preventive Medications',
    'CGRP monoclonal antibodies': 'Migraine Preventive Medications',
    'Gepants': 'Migraine Preventive Medications',
    'Triptans': 'Migraine Acute Medications',
    'Gepants': 'Migraine Acute Medications',  # Note that Gepants appears in both Preventive and Acute
    'Ditan': 'Migraine Acute Medications',
    'Ergots': 'Migraine Acute Medications'
}

# Lowercase everything!
medication_keywords = [x.lower() for x in medication_keywords]
med_group_mapping = dict((k.lower(), v.lower()) for k, v in med_group_mapping.items())
medication_type_mapping = dict((k.lower(), v.lower()) for k, v in medication_type_mapping.items())

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

# Function to classify as preventive or acute medication
def classify_medication_type(groups):
    if groups:
        types = {medication_type_mapping[group] for group in groups.split(', ') if group in medication_type_mapping}
        return ', '.join(types)
    return ''

df['text'] = df['text'].str.lower()  # Convert entire text column to lowercase
df['sentiment_score'] = df['text'].apply(sentiment_score)
df['matched_keywords'] = df['text'].apply(find_keywords)
df['med_group'] = df['matched_keywords'].apply(map_to_med_group)
df['prev_acute'] = df['med_group'].apply(classify_medication_type)

df.to_csv('results/migraine_med_sentiments.csv', index=False)

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
keyword_counts = count_keywords(df['matched_keywords'])
keyword_counts = {key: val for key, val in keyword_counts.items() if val > 0}
# print(keyword_counts)

plt.figure(figsize=(12, 8))
plt.bar(keyword_counts.keys(), keyword_counts.values(), color='blue')
plt.xlabel('Medication Keywords', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Frequency of Each Medication Keyword in Texts', fontsize=16)
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.grid(True)
plt.savefig('figures/freq_medication_keywords.png')

med_group = count_keywords(df['med_group'])
# print(med_group)

plt.figure(figsize=(12, 8))
plt.bar(med_group.keys(), med_group.values(), color='blue')
plt.xlabel('Medication Groups', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Frequency of Each Medication Group in Texts', fontsize=16)
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
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
df_keywords = expand_rows(df, 'matched_keywords')
df_med_groups = expand_rows(df, 'med_group')
df_prev_acute = expand_rows(df, 'prev_acute')

# Strip extra whitespace
df_keywords['matched_keywords'] = df_keywords['matched_keywords'].str.strip()
df_keywords = df_keywords[df_keywords['matched_keywords'] != '']

df_med_groups['med_group'] = df_med_groups['med_group'].str.strip()
df_med_groups = df_med_groups[df_med_groups['med_group'] != '']

df_prev_acute['prev_acute'] = df_prev_acute['prev_acute'].str.strip()
df_prev_acute = df_prev_acute[df_prev_acute['prev_acute'] != '']

plt.figure(figsize=(18, 14))
sns.boxplot(y='matched_keywords', x='sentiment_score', data=df_keywords, orient='h', palette='coolwarm')
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