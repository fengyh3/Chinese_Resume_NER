from data import build_corpus
from evaluate import Metrics
from models.hmm import HMMModel
from models.crf import CRFModel
from models.lstm import BiLSTM
from models.BiLSTM_CRF import BiLSTM_CRF
from utils import *


def main():

    print('读取数据...')
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus('train')
    dev_word_lists, dev_tag_lists = build_corpus('dev', maek_vocab = False)
    test_word_lists, test_tag_lists = build_corpus('test', maek_vocab = False)

    print('训练HMM模型...')
    hmm_model = HMMModel(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    pred_tag_lists = hmm_model.test(test_word_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores()

    print('训练CRF模型...')
    crf_model = CRFModel(max_iterations = 90)
    crf_model.train(train_word_lists, train_tag_lists)
    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists)
    metrics.report_scores()
    
    
    print('训练BiLSTM模型...')
    word2id, tag2id = extend_maps(word2id, tag2id)
    bilstm = BiLSTM(len(word2id), len(tag2id))
    bilstm.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id, 0.8)
    bilstm.dev_test(test_word_lists, test_tag_lists, word2id, tag2id)
    bilstm.close_sess()
    

    print('训练BiLSTM-CRF模型...')
    bilstm_crf = BiLSTM_CRF(len(word2id), len(tag2id))
    bilstm_crf.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id, 0.8)
    bilstm_crf.dev_test(test_word_lists, test_tag_lists, word2id, tag2id)
    bilstm_crf.close_sess()


if __name__ == "__main__":
    main()
