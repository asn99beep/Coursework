'''
we will use ETL to analyze data from a CSV file on TV show South Park dialogs.
using textblob we will analyze the dialogs sentiment to determine which character 
is the most negative. because of the structure of the CSV file, we will need to combine 
two columns of dialog. we will then load our findings into a text file. in addition, we will
analyze overall which character has the most dialog lines, this information will be 
visualized using a bar chart with matplotlib. 
'''

import matplotlib.pyplot as plt
import textblob
import numpy as np
import pandas as pd
import pathlib as pb

# Extract: load the CSV file and read the data (https://www.kaggle.com/datasets/tovarischsukhov/southparklines/discussion?sort=undefined)
data = pd.read_csv(r'C:\Users\aaliy\Downloads\southpark_dialogs.csv')
characters = data['Character'].unique()
sentiment_dict = {}

# Our CSV file has two columns of dialog so we need to combine them
def combine_dialogs(row):
    return f"{row['Line1']} {row['Line2']}"

# Transform: analyze sentiment for each character
for character in characters:
    character_dialogs = data[data['Character'] == character]['Line']
    total_sentiment = 0
    for dialog in character_dialogs:
        blob = textblob.TextBlob(dialog)
        total_sentiment += blob.sentiment.polarity
    average_sentiment = total_sentiment / len(character_dialogs)
    sentiment_dict[character] = average_sentiment

# Determine the most negative character
most_negative_character = min(sentiment_dict, key=sentiment_dict.get)

# Load: write the findings to a text file 
findings_path = pb.Path(f"{pb.Path.cwd()}/south_park_sentiment_findings.txt")
findings_path.touch()
with findings_path.open(mode="w", encoding="utf-8") as file:
    file.write(f"The most negative character is {most_negative_character} with an average sentiment of {sentiment_dict[most_negative_character]:.2f}\n")

# Transform: analyze which character has the most dialog lines
# because there are so nany characters with few lines, we will only visualize the top 10 characters
if 'Line' in data.columns:
    line_count = data['Character'].value_counts()
else:
    data['Combined_Line'] = data.apply(combine_dialogs, axis=1)
    line_count = data['Character'].value_counts()
top_10_line_count = line_count.head(10)

# Load: create a bar chart of the line counts
plt.figure(figsize=(10, 6))
top_10_line_count.plot(kind='bar')
plt.xlabel('Characters')
plt.ylabel('Number of Dialog Lines')
plt.title('Top 10 South Park Characters Dialog Line Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('southpark_top10_linecounts.png')
plt.show()




