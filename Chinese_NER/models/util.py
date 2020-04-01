import numpy as np

def get_next_batch(x, y, word2id, tag2id, batch_size):
    choose = np.random.randint(0, len(x), batch_size)
    m = 0
    for i in range(batch_size):
        m = max(m, len(x[choose[i]]))

    batch_x = np.ones((batch_size, m)).astype(np.int32) * word2id['<pad>']
    batch_y = np.ones((batch_size, m)).astype(np.int32) * tag2id['<pad>']
    seq_len = np.zeros((batch_size)).astype(np.int32)
    for i in range(batch_size):
        seq_len[i] = len(x[choose[i]])
        for j in range(seq_len[i]):
            batch_x[i][j] = word2id.get(x[choose[i]][j], word2id['<unk>'])
            batch_y[i][j] = tag2id.get(y[choose[i]][j], tag2id['<unk>'])

    return batch_x, batch_y, seq_len


def get_batches(x, y, word2id, tag2id, batch_size):
    #按序列长度排序
    pairs = list(zip(x, y))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]

    x, y = list(zip(*pairs))

    batches_x = []
    batches_y = []
    batches_seq_len = []

    for b in range(int(len(x) / batch_size)):
        max_size = 0
        for k in range(batch_size):
            max_size = max(max_size, len(x[b * batch_size + k]))
        batch_x = np.ones((batch_size, max_size)).astype(np.int32) * word2id['<pad>']
        batch_y = np.ones((batch_size, max_size)).astype(np.int32) * tag2id['<pad>']
        seq_len = np.zeros((batch_size)).astype(np.int32)
        for i in range(batch_size):
            seq_len[i] = len(x[b * batch_size + i])
            for j in range(seq_len[i]):
                batch_x[i][j] = word2id.get(x[b * batch_size + i][j], word2id['<unk>'])
                batch_y[i][j] = tag2id.get(y[b * batch_size + i][j], tag2id['<unk>'])
        batches_x.append(batch_x)
        batches_y.append(batch_y)
        batches_seq_len.append(seq_len)

    remain = int(len(x) / batch_size) * batch_size
    max_size = 0
    for k in range(len(x) % batch_size):
        max_size = max(max_size, len(x[remain + k]))
    batch_x = np.ones((len(x) % batch_size, max_size)).astype(np.int32) * word2id['<pad>']
    batch_y = np.ones((len(x) % batch_size, max_size)).astype(np.int32) * tag2id['<pad>']
    seq_len = np.zeros((len(x) % batch_size)).astype(np.int32)
    for i in range(len(x) % batch_size):
        seq_len[i] = len(x[remain + i])
        for j in range(seq_len[i]):
            batch_x[i][j] = word2id.get(x[remain + i][j], word2id['<unk>'])
            batch_y[i][j] = tag2id.get(y[remain + i][j], tag2id['<unk>'])
    batches_x.append(batch_x)
    batches_y.append(batch_y)
    batches_seq_len.append(seq_len)

    return batches_x, batches_y, batches_seq_len