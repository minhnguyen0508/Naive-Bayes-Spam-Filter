import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Convert labels if labeled spam = 1, else = 0
df['spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data to fit the machine requirement 
x_train, x_test, y_train, y_test = train_test_split(
    df.text, df.spam, test_size=0.25
)

# Vectorize text to BOW matrix
vectorizer = CountVectorizer()
x_train_count_sparse = vectorizer.fit_transform(x_train.values)

# GaussianNB requires dense data — convert here
x_train_count = x_train_count_sparse.toarray()

# Train GaussianNB
model = GaussianNB()
model.fit(x_train_count, y_train)

# Testing on example input
with open("test_email.txt", "r") as email:
    email_spam_count_sparse = vectorizer.transform(email)

#Convert test vector to dense 
email_spam_count = email_spam_count_sparse.toarray()

# Make prediction
spam_ham_value = model.predict(email_spam_count)
print("Scam/Ham value of each lines: ")
print(spam_ham_value) #show point value (1 = scam, 0 = ham)

#tally up the point, if more spam pts then email is scam
spam_pot = np.sum(spam_ham_value)   
ham_pot = len(spam_ham_value) - spam_pot
print("Verdict: ")
if spam_pot > ham_pot:
    print("This email is a scam!")
else:
    print("This email is safe!")
