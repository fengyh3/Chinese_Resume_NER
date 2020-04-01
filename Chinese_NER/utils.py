

def flatten_lists(lists):
    flatten_list = []
    for li in lists:
        if type(li) == list:
            flatten_list += li
        else:
            flatten_list.append(li)
    return flatten_list

def extend_maps(word2id, tag2id):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    return word2id, tag2id

def merge_maps(dict1, dict2):
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1