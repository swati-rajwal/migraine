'''
### A.   Commonly used migraine preventive medications, generic (US brand name): 
•        Topiramate (Topamax)
•        Propranolol (Inderal)
•        Amitriptyline (Elavil)
•        OnabotulinumtoxinA (Botox)
•        CGRP monoclonal antibodies: combine the 4- erenumab (Aimovig), galcanezumab (Emgality), fremanezumab (Ajovy), eptinezumab (Vyepti)
•        Gepants: Atogepant (Qulipta),  rimegepant (Nurtec),
(A thought, since the number of Botox mentioned is so high, perhaps some include migraine users talking about Botox for non-migraine indication, such as cosmetics..etc, this would be a limitation)

### B.   Commonly used migraine acute medications: 
• Triptans: Sumatriptan (Imitrex), Rizatriptan (Maxalt), Eletriptan (Relpax), Naratriptan (Amerge), Frovatriptan (Frova), Zolmitriptan (Zomig), Almotriptan (Axert), 
• Gepants: ubrogepant (Ubrelvy), rimegepant (Nurtec), zavegepant (Zavzpret)
• Ditan: lasmiditan (Reyvow)
• [Not sure if we included this category the first round] Ergots: Dihydroergotamine (DHE, Migranal, Trudhesa), ergotamine (Cafergot)
'''
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('data/B_processed_dataset/combined_dataset.csv')

lengths1 = df['text'].str.len()
plt.figure(figsize=(10, 6))  # Adjust the size of your plot
plt.hist(lengths1, bins=50, alpha=0.5, density=True)
mean_length1 = lengths1.mean()
plt.axvline(mean_length1, color='red', linestyle='dashed', linewidth=1, label=f'Mean text length: {mean_length1:.2f}')
plt.xlabel('Length of Clinical Text')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig('figures/lengths.png')

medication_keywords = ['Topiramate','Topamax','Propranolol','Inderal', 'Atenolol','tenormin', 'Metoprolol','Toprol','Amitriptyline','Elavil','Nortriptyline', 'Pamelor','OnabotulinumtoxinA','Botox','erenumab','Aimovig', 'galcanezumab','Emgality', 'fremanezumab','Ajovy','eptinezumab','Vyepti','Atogepant','Qulipta', 'ubrogepant','Ubrelvy', 'rimegepant','Nurtec',
'Sumatriptan','Imitrex', 'Rizatriptan','Maxalt', 'Eletriptan','Relpax', 'Naratriptan','Amerge', 'Frovatriptan','Frova', 'Zolmitriptan','Zomig', 'Almotriptan','Axert',
'Lasmiditan','Reyvow','Dihydroergotamine','DHE','Migranal', 'ergotamine',]

medication_keywords = [x.lower() for x in medication_keywords]

sentences = df['text']
sentiment=[]
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    sent=vs['compound']
    sentiment.append(sent)

df['sentiment score'] = sentiment  # Store score of each reddit text in new column

Brand_name_Dict = {'Topiramate':'Topamax','Topamax':'Topamax','Propranolol':'Inderal','Inderal':'Inderal','Atenolol':'tenormin','tenormin':'tenormin','Metoprolol':'Toprol','Toprol':'Toprol','Amitriptyline':'Elavil','Elavil':'Elavil','Nortriptyline':'Pamelor','Pamelor':'Pamelor','OnabotulinumtoxinA':'Botox','Botox':'Botox','erenumab':'Aimovig','Aimovig':'Aimovig','galcanezumab':'Emgality','Emgality':'Emgality','fremanezumab':'Ajovy','Ajovy':'Ajovy','eptinezumab':'Vyepti','Vyepti':'Vyepti','Atogepant':'Qulipta','Qulipta':'Qulipta','ubrogepant':'Ubrelvy','Ubrelvy':'Ubrelvy','rimegepant':'Nurtec','Nurtec':'Nurtec','Sumatriptan':'Imitrex','Imitrex':'Imitrex', 'Rizatriptan':'Maxalt','Maxalt':'Maxalt','Eletriptan':'Relpax','Relpax':'Relpax','Naratriptan':'Amerge','Amerge':'Amerge','Frovatriptan':'Frova','Frova':'Frova','Zolmitriptan':'Zomig','Zomig':'Zomig','Almotriptan':'Axert','Axert':'Axert','Lasmiditan':'Reyvow','Reyvow':'Reyvow','Dihydroergotamine':'DHE','DHE':'DHE','Dihydroergotamine':'Migranal','Migranal':'Migranal','ergotamine':'ergotamine','ergotamine':'Ergot','Ergot':'Ergot'}
Brand_name_Dict = dict((k.lower(), v.lower()) for k, v in Brand_name_Dict.items())   # Converting to lower letter for easy comparision

Med_Group_Dict = {'Topiramate':'Topiramate (Topamax)','Topamax':'Topiramate (Topamax)','Propranolol':'Beta Blockers','Inderal':'Beta Blockers','tenormin':'Beta Blockers','Toprol':'Beta Blockers','Atenolol':'Beta Blockers','Metoprolol':'Beta Blockers','Amitriptyline':'Tricyclic antidepressants','Elavil':'Tricyclic antidepressants','Nortriptyline':'Tricyclic antidepressants','Pamelor':'Tricyclic antidepressants','OnabotulinumtoxinA':'OnabotulinumtoxinA (Botox)','Botox':'OnabotulinumtoxinA (Botox)','erenumab':'CGRP monoclonal antibodies','Aimovig':'CGRP monoclonal antibodies','galcanezumab':'CGRP monoclonal antibodies','Emgality':'CGRP monoclonal antibodies','fremanezumab':'CGRP monoclonal antibodies','Ajovy':'CGRP monoclonal antibodies','eptinezumab':'CGRP monoclonal antibodies','Vyepti':'CGRP monoclonal antibodies','Atogepant':'Gepants','Qulipta':'Gepants','Ubrelvy':'Gepants','Nurtec':'Gepants','ubrogepant':'Gepants','rimegepant':'Gepants','Imitrex':'Triptans','Maxalt':'Triptans','Relpax':'Triptans','Amerge':'Triptans','Frova':'Triptans','Zomig':'Triptans','Axert':'Triptans','Sumatriptan':'Triptans','Rizatriptan':'Triptans','Eletriptan':'Triptans','Naratriptan':'Triptans','Frovatriptan':'Triptans','Zolmitriptan':'Triptans','Almotriptan':'Triptans','Reyvow':'Lasmiditan (Reyvow)','Lasmiditan':'Lasmiditan (Reyvow)','Dihydroergotamine':' Dihydroergotamine (DHE)','ergotamine':'Dihydroergotamine (DHE)'}
Med_Group_Dict = dict((k.lower(), v.lower()) for k, v in Med_Group_Dict.items())   # Converting to lower letter for easy comparision

df['medication_matched'] = ''
df['brand_name'] = ''
df['med_group'] = ''

# To store all sentiment scores for each medication group
senti_score_dict = {'Topiramate (Topamax)':[], 'Beta Blockers':[], 'Tricyclic antidepressants':[], 'OnabotulinumtoxinA (Botox)':[], 'CGRP monoclonal antibodies':[], 'Gepants':[], 'Triptans':[], 'Lasmiditan (Reyvow)':[], 'Dihydroergotamine (DHE)':[]}
senti_score_dict = dict((k.lower(), v) for k, v in senti_score_dict.items())   # Converting to lower letter for easy comparision

# To track frequency of each medication group
frequency_dict = {'Topiramate (Topamax)':0, 'Beta Blockers':0, 'Tricyclic antidepressants':0, 'OnabotulinumtoxinA (Botox)':0, 'CGRP monoclonal antibodies':0, 'Gepants':0, 'Triptans':0, 'Lasmiditan (Reyvow)':0, 'Dihydroergotamine (DHE)':0}
frequency_dict = dict((k.lower(), v) for k, v in frequency_dict.items())   # Converting to lower letter for easy comparision

for i, row in df.iterrows():
  curr_sentence = row['text'].lower().split()   # Convert str into list of words
  matches = list(set(curr_sentence).intersection(set(medication_keywords)))
  
  if(matches):
    df.at[i,'medication_matched'] = matches
    brand_name = []  
    group_name = []
    for j in range(len(matches)):
      dict_brand_value = Brand_name_Dict.get(matches[j])
      dict_group_value = Med_Group_Dict.get(matches[j])
      if(dict_brand_value!=None):
        brand_name.append(dict_brand_value)  

      if(dict_group_value!=None):
        group_name.append(dict_group_value)
        
        frequency_dict[dict_group_value] = frequency_dict[dict_group_value]+1  # Update the frequency
        senti_score_dict[dict_group_value].append(row['sentiment score'])

    df.at[i,'brand_name'] = list(set(brand_name))
    df.at[i,'med_group'] = list(set(group_name))

df.to_csv('results/sentiment_results.csv', index=False)

statistic_values = pd.DataFrame(columns=['group_name','frequency','mean','median','Standard_deviation'])
statistic_values['group_name'] = frequency_dict.keys()
statistic_values['sentiment_scores'] = list(senti_score_dict.values())
statistic_values['frequency'] = frequency_dict.values()

mean_list = []
median_list = []
sd_list = []
senti_scores = list(senti_score_dict.values())
for i in range(len(senti_scores)):
  mean_list.append(round(np.mean(senti_scores[i]),2))
  median_list.append(round(np.median(senti_scores[i]),2))
  sd_list.append(round(np.std(senti_scores[i]),2))

statistic_values['mean'] = mean_list
statistic_values['median'] = median_list
statistic_values['Standard_deviation'] = sd_list
# statistic_values.fillna(0)
# print(statistic_values)
statistic_values.to_csv('results/sentiment_statistics.csv',index=False)

plt_1 = plt.figure(figsize=(9, 5))

selected = ['Topiramate (Topamax)','Beta Blockers','Tricyclic antidepressants','OnabotulinumtoxinA (Botox)','CGRP monoclonal antibodies','Gepants','Triptans','Lasmiditan (Reyvow)','Dihydroergotamine (DHE)']
selected = [x.lower() for x in selected]    #All in lower letters
for name in selected:
    temp_df = statistic_values[statistic_values['group_name']==name].reset_index(drop=True)
    scores = temp_df['sentiment_scores'][0]
    sns.kdeplot(scores, label=name, clip=[-1, 1])
plt.title('Kernel Density Estimation of Sentiment Scores across Medication Groups')
plt.xlabel('Sentiment score')
plt.legend()
plt.grid()
plt.savefig('figures/sentiment_distribution.png')

colors = plt.cm.jet(np.linspace(0, 1, len(statistic_values)))
plt.figure(figsize=(10, 6))
for i, row in statistic_values.iterrows():
    plt.scatter(row['mean'], row['median'], color=colors[i], label=row['group_name'])

plt.title('Sentiment Score Distribution per Medication Group')
plt.xlabel('Mean Sentiment Score')
plt.ylabel('Median Sentiment Score')
plt.grid(True)
plt.legend(title='Group Names', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjusts plot to make room for the legend
plt.grid(True)
plt.savefig('figures/mean_median_sentiment_plot.png')