def train_naive_bayes(train_data):
    ham_data, spam_data, train_samples, labels = split_by_label(train_data)

    ham_vocab = get_vocab(ham_data)
    spam_vocab = get_vocab(spam_data)
    vocabulary = get_vocab(train_samples)
    ham_word_probs = {word:0 for word in vocabulary}
    spam_word_probs = {word:0 for word in vocabulary}
    # word_count
    model = {'p_ham':len(ham_data)/len(train_samples),
             'p_spam':len(spam_data)/len(train_samples),
             'ham_word_probs':ham_word_probs,
             'spam_word_probs':spam_word_probs}
    print(model)

def test_naive_bayes(test_data, model):
    processed_data = process_dataset(test_data)
    num_correct = 0
    for document, label in processed_data:
        # out_label = 'ham' if label else 'spam' #debug
        # print(f"{document}, {out_label}") #debug
        doc_p_ham = model['p_ham']
        doc_p_spam = model['p_prob']
        for word in document:
            if word in model['ham_word_probs']:
                doc_p_ham *= model['ham_word_probs'][word]
            if word in model['spam_word_probs']:
                doc_p_spam *= model['spam_word_probs'][word]
            num_correct += (doc_p_ham > doc_p_spam) == label
    return num_correct / len(test_data)

def process_dataset(dataset):
    return dataset

def get_vocab(dataset):
    vocab = set({})
    for document in dataset:
        print(document)
        for word, _ in document.items():
            vocab.update([word])
    return vocab

def split_by_label(dataset):
    ham_data = []
    spam_data = []
    samples = []
    labels = []
    for sample in dataset:
        if sample[1] == 1: ham_data.append(sample[0])
        if sample[1] == 0: spam_data.append(sample[0])
        samples.append(sample[0])
        labels.append(sample[1])
    return ham_data, spam_data, samples, labels

if __name__ == "__main__":
    data = [({'hello': 1, 'this':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'hello': 1, 'welcome':1, 'to':1, 'my':1, 'spam':1}, 0)]
    model = None
    train_naive_bayes(data)
    # test_naive_bayes(data, model)