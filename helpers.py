from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import os

def load_pos_data(fname):
    file_data = pd.read_csv(fname, sep=" ", header=None)
    file_data[0] = file_data[0].str.lower()
    unique_counts = file_data[0].value_counts()
    filtered = unique_counts.drop(unique_counts[unique_counts==1].index)
    filtered = filtered.drop(filtered[filtered==2].index)
    filtered = filtered.drop(filtered[filtered==3].index)
    filtered = filtered.drop([',','the','.','to','of','a','and','in','\'s','for','that', 'is'])
    filtered = filtered.to_frame()
    filtered['tag'] = None
    filtered['label'] = None
    # print(filtered)
    # for row in filtered.iterrows():
    #     row[1] = 
    # words_and_labels = filtered.join(file_data)
    # print(words_and_labels)
    return file_data


def load_spam_data(dirname):
    raw_ham_data = []
    raw_spam_data = []

    ham_dir = os.path.join(dirname, 'ham')
    spam_dir = os.path.join(dirname, 'spam')
    ham_files = os.listdir(os.path.join(dirname, 'ham'))
    spam_files = os.listdir(os.path.join(dirname, 'spam'))

    ham_emails = []
    spam_emails = []

    for file in range(5):
        with open(os.path.join(ham_dir, ham_files[file]), 'r') as file:
            email = file.read()
            ham_emails.append((word_to_features(word_tokenize(email)), 1))
    for file in range(5):
        with open(os.path.join(spam_dir, spam_files[file]), 'r') as file:
            email = file.read()
            spam_emails.append((word_to_features(word_tokenize(email)), 0))
    combined_emails = ham_emails + spam_emails
    np.random.shuffle(combined_emails)
    train = combined_emails[int(len(combined_emails)*0.8):]
    test = combined_emails[:int(len(combined_emails)*0.8)]
    return train, test

def word_to_features(line):
    return dict([word, 1] for word in line)

if __name__ == "__main__":
    # load_pos_data("pos_test.txt")
    load_spam_data("enron1")