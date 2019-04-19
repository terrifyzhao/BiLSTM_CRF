from data_utils import pad_sequences, shuffle


def load_vocab():
    vocab = [str(line).split()[0] for line in open('input/vocab.txt', encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def load_tag():
    vocab = [str(line).split()[0] for line in open('input/tags.txt', encoding='utf-8').readlines()]
    tag2idx = {word: index for index, word in enumerate(vocab)}
    idx2tag = {index: word for index, word in enumerate(vocab)}
    return tag2idx, idx2tag


def load_data():
    with open('input/data.txt', encoding='utf-8') as file:
        X = []
        y = []
        seq_len = []

        X, y = shuffle(X, y)
        X = X.tolist()
        y = y.tolist()

        for index, lien in enumerate(file.readlines()):
            if index % 2 == 0:
                X.append(lien.strip())
            else:
                y.append(lien.strip())
        word2idx, _ = load_vocab()
        x_index = []
        for sentence in X:
            sentence = list(sentence.replace(' ', ''))
            if len(sentence) < 15:
                seq_len.append(len(sentence))
            else:
                seq_len.append(15)
            # max_len = max(seq_len)
            sentence_index = [word2idx[i] if i in list(word2idx.keys()) else 1 for i in sentence]
            x_index.append(sentence_index)
        X = pad_sequences(x_index, maxlen=15, value=0)

        tag2idx, _ = load_tag()
        tag_indexs = []
        for tag in y:
            tag = tag.split(' ')
            tag_index = [tag2idx[i] if i in list(tag2idx.keys()) else 0 for i in tag]
            tag_indexs.append(tag_index)
        y = pad_sequences(tag_indexs, maxlen=15, value=0)

        return X, y, seq_len


def train_data():
    X, y, seq_len = load_data()
    data_len = len(X)
    end = int(0.8 * data_len)
    return X[0:end], y[0:end], seq_len[0:end]


def eval_data():
    X, y, seq_len = load_data()
    data_len = len(X)
    end = int(0.8 * data_len)
    return X[end:], y[end:], seq_len[end:]


if __name__ == '__main__':
    load_data()
