import numpy as np

class HMMModel(object):
    def __init__(self, state_size, observe_size):
        self.state_size = state_size
        self.observe_size = observe_size

        #状态转移矩阵，state[i][j]表示从状态i转移到状态j的概率
        self.state = np.zeros((state_size, state_size))
        #观测概率矩阵，observe[i][j]表示状态i下生成观测j的概率
        self.observe = np.zeros((state_size, observe_size))
        #初始状态概率
        self.Pi = np.zeros((state_size))

    def train(self, word_lists, tag_lists, word2id, tag2id):
        '''
        采用极大似然估计来估计参数矩阵
        '''
        #可能读取数据出错，或者数据集有污染
        assert len(tag_lists) == len(word_lists)

        #估计转移概率矩阵
        for tag_list in tag_lists:
            for i in range(len(tag_list) - 1):
                cur_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.state[cur_tagid][next_tagid] += 1

        #需要做一个平滑，解决频数为0的情况
        self.state[self.state == 0] = 1e-10
        self.state = self.state / self.state.sum(axis = 1, keepdims = True)

        #估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.observe[tag_id][word_id] += 1

        #依然是要做平滑
        self.observe[self.observe == 0] = 1e-10
        self.observe = self.observe / self.observe.sum(axis = 1, keepdims = True)

        #估计初始状态矩阵
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def viterbi_decoding(self, word_list, word2id, tag2id):
        #需要解决概率相乘造成的数据下溢的问题
        state = np.log(self.state)
        observe = np.log(self.observe)
        Pi = np.log(self.Pi)

        #使用动态规划来寻找路径最大值
        #其中viterbi[i, j]表示序列第j个的状态是i_0,i_1,....,i_i的概率
        seq_len = len(word_list)
        viterbi = np.zeros((self.state_size, seq_len))
        #backtract是用来回溯找路径的
        backtract = np.zeros((self.state_size, seq_len))

        #首先是第一步，dp的开始
        #observe_T是观测矩阵的转置，所以observe_T[i]表示word是i的所有状态的概率
        start_wordid = word2id.get(word_list[0], None)
        observe_T = observe.T
        if start_wordid is None:
            state_tmp = np.log(np.ones(self.state_size) / self.state_size)
        else :
            state_tmp = observe_T[start_wordid]

        viterbi[:, 0] = Pi + state_tmp
        backtract[:, 0] = -1

        #动规状态转移公式
        #viterbi[tag_id, step] = max(viterbi[:, step - 1] * state.T[tag_id] * observe_T[word])
        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            if wordid is None:
                state_tmp = np.log(np.ones(self.state_size) / self.state_size)
            else :
                state_tmp = observe_T[wordid]
            for tag_id in range(len(tag2id)):
                #因为取了log，就变成了+
                tmp = viterbi[:, step - 1] + state[:, tag_id]
                max_prob = np.max(tmp, axis = 0)
                max_id = np.argmax(tmp, axis = 0)
                viterbi[tag_id, step] = max_prob + state_tmp[tag_id]
                backtract[tag_id, step] = max_id

        best_prob = np.max(viterbi[:, seq_len - 1], axis=0)
        last_path = np.argmax(viterbi[:, seq_len - 1], axis=0)

        #回溯找路径
        best_path = [last_path, ]
        for cur_step in range(seq_len - 1, 0, -1):
            last_path = int(backtract[last_path][cur_step])
            best_path.append(last_path)

        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list

    def test(self, word_lisst, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lisst:
            pred_tag_lists.append(self.viterbi_decoding(word_list, word2id, tag2id))
        return pred_tag_lists