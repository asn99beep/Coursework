# this program will use package TextBlob to analyze the sentiment 
# subjectivity in user inputted sentence to return an average
# we will also usse user input to create a dictionary of words and their parts of speech 
# using a user inputted sentence

from textblob import TextBlob
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


words_list = []

# Get user input and calculate average subjectivity
user_words = input("Please enter a sentence: ")
words_list = [word.strip() for word in user_words.split(",")]
total_subjectivity = 0
for word in words_list:
    blob = TextBlob(word)
    total_subjectivity += blob.sentiment.subjectivity
average_subjectivity = total_subjectivity / len(words_list)
print(words_list)

# Determine if the average subjectivity indicates objective or subjective language
if average_subjectivity < 0.5:
    print(f"Your sentence is objective, scoring a: {average_subjectivity} averge subjectivity.")
else:
    print(f"Your sentence is subjective, scoring a: {average_subjectivity} averge subjectivity.")
   

# Create an empty dictionary to store words and their parts of speech
pos_dict = {}

while True:
    user_sentence = input("Please enter a sentence to assign parts of speech (or type 'exit' to quit): ")
    if user_sentence.lower() == 'exit':
        break
    blob = TextBlob(user_sentence)
    pos_tags = blob.tags
    for word, pos in pos_tags:
        pos_dict[word] = pos
    print(f"Parts of Speech for the sentence:")
    for word, pos in pos_tags:
        print(f"- {word}: {pos}")
print(pos_dict)
