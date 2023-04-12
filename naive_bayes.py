def train_naive_bayes(train_data):
    ham_data, spam_data, train_samples, labels = split_by_label(train_data)
    n_words = count_words(train_samples)
    n_ham_words = count_words(ham_data)
    n_spam_words = count_words(spam_data)
    ham_vocab = get_vocab(ham_data)
    spam_vocab = get_vocab(spam_data)
    vocabulary = get_vocab(train_samples)
    ham_word_probs = {word:0 for word in vocabulary}
    spam_word_probs = {word:0 for word in vocabulary}
    for index, document in enumerate(train_samples):
        doc_label = labels[index]
        for key, _ in document.items():
            if doc_label == 1: ham_word_probs[key] += 1+1/n_words
            else: spam_word_probs[key] += 1/n_words
    model = {'p_ham':n_ham_words/n_words,
             'p_spam':n_spam_words/n_words,
             'ham_word_probs':ham_word_probs,
             'spam_word_probs':spam_word_probs}
    return model

def test_naive_bayes(test_data, model):
    processed_data = process_dataset(test_data)
    num_correct = 0
    for document, label in processed_data:
        # out_label = 'ham' if label else 'spam' #debug
        # print(f"{document}, {out_label}") #debug
        doc_p_ham = model['p_ham']
        doc_p_spam = model['p_spam']
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
        for word, _ in document.items():
            vocab.update([word])
    return vocab

def count_words(dataset):
    word_count = 0
    for document in dataset:
        for word, _ in document.items():
            word_count += 1
    return word_count

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
    test_data = [({'good': 1, 'this':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'pepperoni': 1, 'welcome':1, 'to':1, 'my':1, 'spam':1}, 0),
            ({'hello': 1, 'apples':1, 'is':1, 'my':1, 'test':1}, 1),
            ({'welcome': 1, 'this':1, 'welcome':1, 'my':1, 'test':1}, 0)]
    model = train_naive_bayes(data)
    print(f"Accuracy: {test_naive_bayes(test_data, model)}")