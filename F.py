import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results/sentiment_results.csv')
plt.figure(figsize=(10, 6))
plt.scatter(df['label'], df['sentiment score'], alpha=0.5)
plt.title('Scatter Plot of Label vs Sentiment Score')
plt.xlabel('Label')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.savefig('figures/correlation.png')
