from os.path import join

def build_corpus(dataset, maek_vocab = True, data_dir = "./data"):
    word_lists = []
    tag_lists = []
    with open(join(data_dir, dataset + ".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    if maek_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)

        return word_lists, tag_lists, word2id, tag2id

    return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for l in lists:
        for word in l:
            if word not in maps:
                maps[word] = len(maps)

    return maps