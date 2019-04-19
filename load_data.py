from tflearn.data_utils import pad_sequences, shuffle


def load_vocab():
    vocab = [str(line).split()[0] for line in open('input/vocab.txt').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def load_tag():
    vocab = [str(line).split()[0] for line in open('input/tags.txt').readlines()]
    tag2idx = {word: index for index, word in enumerate(vocab)}
    idx2tag = {index: word for index, word in enumerate(vocab)}
    return tag2idx, idx2tag


def load_data():
    with open('data/data.txt') as file:
        X = []
        y = []
        for index, lien in enumerate(file.readlines()):
            if index % 2 == 0:
                X.append(lien.strip())
            else:
                y.append(lien.strip())
        word2idx, _ = load_vocab()
        x_index = []
        for sentence in X:
            sentence = list(sentence.replace(' ', ''))
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

        return X, y


if __name__ == '__main__':
    load_data()
