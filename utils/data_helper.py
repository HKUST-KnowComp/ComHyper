from .loader import read_sparse_matrix 
from .util import load_phrase_word2vec
from .util import load_gensim_word2vec

import torch
import numpy as np
import scipy.sparse as sparse

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, config, svd=False, train=True):
        # generate ppmi matrix for co-occurence
        pattern_filename = config.get("data", "pattern_filename")

        k = int(config.getfloat("hyperparameters", "svd_dimension"))
        self.batch_size = int(config.getfloat("hyperparameters", "batch_size"))
        self.negative_num = int(config.getfloat("hyperparameters", "negative_num"))

        csr_m, self.id2word, self.vocab, _ = read_sparse_matrix(
            pattern_filename, same_vocab=True)

        self.word2id = {}
        for i in range(len(self.id2word)):
            self.word2id[self.id2word[i]] = i 

        self.matrix = csr_m.todok()
        self.p_w = csr_m.sum(axis=1).A[:,0]
        self.p_c = csr_m.sum(axis=0).A[0,:]
        self.N = self.p_w.sum() 

        # for w2v
        if train:
            #self.wordvecs = load_phrase_word2vec("/home/shared/acl-data/embedding/ukwac.model", self.vocab)
            self.wordvecs = load_gensim_word2vec("/home/shared/acl-data/embedding/ukwac.model", self.vocab)
            #print(self.wordvecs["united_states"])

            self.wordvec_weights = self.build_emb()

        tr_matrix = sparse.dok_matrix(self.matrix.shape)
        #print(self.matrix.shape)

        self.left_has = {}
        self.right_has = {}
        for (l,r) in self.matrix.keys():
            pmi_lr = (np.log(self.N) + np.log(self.matrix[(l,r)]) 
                      - np.log(self.p_w[l]) - np.log(self.p_c[r]))

            ppmi_lr = np.clip(pmi_lr, 0.0, 1e12)
            tr_matrix[(l,r)] = ppmi_lr

            if l not in self.left_has:
                self.left_has[l] = []
            self.left_has[l].append(r)
            if r not in self.right_has:
                self.right_has[r] = []
            self.right_has[r].append(l)

        self.ppmi_matrix = tr_matrix

        U, S, V = sparse.linalg.svds(self.ppmi_matrix.tocsr(), k=k)
        self.U = U.dot(np.diag(S))
        self.V = V.T

        if train:
            # self.positive_data, self.positive_label = self.generate_positive()
            self.get_avail_vocab()

    def get_avail_vocab(self):
        avail_vocab = []
        for idx in range(len(self.vocab)):
            if self.id2word[idx] in self.wordvecs:
                avail_vocab.append(idx)
        self.avail_vocab = np.asarray(avail_vocab)
        shuffle_indices_left = np.random.permutation(len(self.avail_vocab))[:20000]
        shuffle_indices_right = np.random.permutation(len(self.avail_vocab))[:20000]
        dev_data = []
        dev_label = []
        self.dev_dict = {}
        for id_case in range(20000):
            id_left = self.avail_vocab[shuffle_indices_left[id_case]]
            id_right = self.avail_vocab[shuffle_indices_right[id_case]]
            dev_data.append([self.w2embid[id_left],self.w2embid[id_right]])
            dev_label.append(self.U[id_left].dot(self.V[id_right]))
            self.dev_dict[(id_left, id_right)] = 1
        self.dev_data = np.asarray(dev_data)
        self.dev_label = np.asarray(dev_label)

    def build_emb(self): 

        tensors = []
        ivocab = []
        self.w2embid = {}
        self.embid2w = {}

        for word in self.wordvecs:
            vec = torch.from_numpy(self.wordvecs[word])
            self.w2embid[self.word2id[word]] = len(ivocab)
            self.embid2w[len(ivocab)] = self.word2id[word]

            ivocab.append(word)
            tensors.append(vec)

        assert len(tensors) == len(ivocab)
        print(len(tensors))
        tensors = torch.cat(tensors).view(len(ivocab), 300)

        return tensors

    def load_vocab(self, w2v_dir, data_dir):
        i2w_path = os.path.join(data_dir, 'ukwac_id2word.pkl')
        w2i_path = os.path.join(data_dir, 'ukwac_word2id.pkl')
        with open(i2w_path, 'rb') as fr:
            self.context_i2w = pickle.load(fr)
        with open(w2i_path, 'rb') as fr:
            self.context_w2i = pickle.load(fr)

        self.PAD = 0
        self.UNK = 1

        # w2v_model = Word2Vec.load(w2v_path)
        # emb = w2v_model.wv
        # oi2ni = {}
        # new_embedding = []
        # new_embedding.append(np.zeros(300))
        # new_embedding.append(np.zeros(300))
        # cnt_ni = 2
        # for _id, word in i2w.items():
        #     if word in emb:
        #         oi2ni[_id] = cnt_ni
        #         cnt_ni += 1 
        #         new_embedding.append(emb[word])
        #     else:
        #         oi2ni[_id] = self.UNK

        oi2ni_path = os.path.join(w2v_dir, 'context_word_oi2ni.pkl')
        w2v_path = os.path.join(w2v_dir, 'context_word_w2v.model.npy')
        with open(oi2ni_path, 'rb') as fr:
            self.context_i2embid = pickle.load(fr)
        self.context_word_emb = np.load(w2v_path)


    def generate_positive(self):

        positive = []
        label = []
        key_list = list(self.ppmi_matrix.keys())
        shuffle_indices = np.random.permutation(len(key_list))

        for shuffle_id in shuffle_indices:
            (l, r) = key_list[shuffle_id]
            if self.id2word[l] in self.wordvecs and self.id2word[r] in self.wordvecs:
                positive.append([self.w2embid[l],self.w2embid[r]])
            # if l in self.context_dict and r in self.context_dict:
                # positive.append([l, r])
                score = self.U[l].dot(self.V[r])
                label.append(score)
                # label.append(self.ppmi_matrix[(l,r)])
        # 119448 positive score 
        positive_train  = np.asarray(positive)[:-2000]

        self.dev_data = np.asarray(positive)[-2000:]
        
        label_train = np.asarray(label)[:-2000]
        self.dev_label = np.asarray(label)[-2000:]

        return positive_train, label_train

    def generate_negative(self, batch_data, negative_num):
        
        negative = []
        label = []

        batch_size = batch_data.shape[0]
    
        for i in range(batch_size):
            # random_idx = np.random.choice(len(self.vocab), 150 , replace=False)
            l = batch_data[i][0]
            l_w = self.embid2w[l]
            r = batch_data[i][1]
            r_w = self.embid2w[r]

            l_neg = l_w
            r_neg = r_w

            num = 0
            for j in range(negative_num):
                left_prob = np.random.binomial(1, 0.5)
                # while True:
                if left_prob:
                    l_neg = np.random.choice(self.avail_vocab, 1)[0]
                else:
                    r_neg = np.random.choice(self.avail_vocab, 1)[0]
                # if (l_neg, r_neg) not in self.matrix.keys() and self.id2word[l_neg] in self.wordvecs and self.id2word[r_neg] in self.wordvecs:
                    # if (l_neg, r_neg) not in self.matrix.keys() and self.l_neg in self.context_dict and self.r_neg in self.context_dict:
                        # break

                negative.append([self.w2embid[l_neg], self.w2embid[r_neg]])
                # negative.append([self.context_dict[l_neg], self.context_dict[r_neg]])
                score = self.U[l_neg].dot(self.V[r_neg])
                # score = 0
                label.append(score)

        negative = np.asarray(negative)
        label = np.asarray(label)
        return negative, label


    def get_batch(self):


        num_positive = len(self.positive_data)

        batch_size = self.batch_size

        if num_positive% batch_size == 0:
            batch_num =  num_positive // batch_size
        else:
            batch_num =  num_positive // batch_size + 1

        shuffle_indices = np.random.permutation(num_positive)

        for batch in range(batch_num):

            start_index = batch * batch_size
            end_index = min((batch+1) * batch_size, num_positive)

            batch_idx = shuffle_indices[start_index:end_index]
    
            batch_positive_data = self.positive_data[batch_idx]
            batch_positive_label = self.positive_label[batch_idx]

            batch_negative_data, batch_negative_label = self.generate_negative(batch_positive_data, self.negative_num)
        
            # batch_positive_data = []
            # for [l, r] in batch_positive_data:
            #     batch_positive_data.append(self.context_dict[l], self.context_dict[r])

            # [batch, 2, doc, 2, seq]
            batch_input = np.concatenate((batch_positive_data, batch_negative_data), axis=0)
            batch_label = np.concatenate((batch_positive_label,batch_negative_label), axis=0)

            yield batch_input, batch_label      

    def sample_batch(self):
        num_data = len(self.avail_vocab)

        batch_size = self.batch_size

        if num_data % batch_size == 0:
            batch_num =  num_data // batch_size
        else:
            batch_num =  num_data // batch_size + 1

        shuffle_indices = np.random.permutation(num_data)

        for batch in range(batch_num):

            start_index = batch * batch_size
            end_index = min((batch+1) * batch_size, num_data)

            batch_idx = shuffle_indices[start_index:end_index]
            batch_data_pair = []
            batch_data_score = []
            batch_data = self.avail_vocab[batch_idx]
  
            for idx_i in batch_data:
                for j in range(self.negative_num):
                    left_prob = np.random.binomial(1, 0.5)
                    if left_prob:
                        while True:
                            idx_j = np.random.choice(self.avail_vocab, 1)[0]
                            if (idx_i, idx_j) not in self.dev_dict:
                                break 
                        batch_data_pair.append([self.w2embid[idx_i], self.w2embid[idx_j]])
                        score = self.U[idx_i].dot(self.V[idx_j])
                    else:
                        while True:
                            idx_j = np.random.choice(self.avail_vocab, 1)[0]
                            if (idx_j, idx_i) not in self.dev_dict:
                                break 
                        batch_data_pair.append([self.w2embid[idx_j], self.w2embid[idx_i]])
                        score = self.U[idx_j].dot(self.V[idx_i])
                    batch_data_score.append(score)
            yield np.asarray(batch_data_pair), np.asarray(batch_data_score)

    def sample_pos_neg_batch(self):
        num_data = len(self.avail_vocab)

        batch_size = self.batch_size

        if num_data % batch_size == 0:
            batch_num =  num_data // batch_size
        else:
            batch_num =  num_data // batch_size + 1

        shuffle_indices = np.random.permutation(num_data)

        for batch in range(batch_num):

            start_index = batch * batch_size
            end_index = min((batch+1) * batch_size, num_data)

            batch_idx = shuffle_indices[start_index:end_index]
            batch_data_pair = []
            batch_data_score = []
            batch_data = self.avail_vocab[batch_idx]
  
            for idx_i in batch_data:
                if idx_i in self.left_has:
                    idx_j_list = np.random.permutation(self.left_has[idx_i])
                    for idx_j in idx_j_list:
                        if idx_j in self.avail_vocab:
                            batch_data_pair.append([self.w2embid[idx_i], self.w2embid[idx_j]])
                            score = self.U[idx_i].dot(self.V[idx_j])
                            batch_data_score.append(score)
                            break

                if idx_i in self.right_has:
                    idx_j_list = np.random.permutation(self.right_has[idx_i])
                    for idx_j in idx_j_list:
                        if idx_j in self.avail_vocab:
                            batch_data_pair.append([self.w2embid[idx_j], self.w2embid[idx_i]])
                            score = self.U[idx_j].dot(self.V[idx_i])
                            batch_data_score.append(score)
                            break
                            
                for j in range(self.negative_num):
                    # left_prob = np.random.binomial(1, 0.5)
                    # if left_prob:
                    while True:
                        idx_j = np.random.choice(self.avail_vocab, 1)[0]
                        if (idx_i, idx_j) not in self.dev_dict:
                            break 
                    batch_data_pair.append([self.w2embid[idx_i], self.w2embid[idx_j]])
                    score = self.U[idx_i].dot(self.V[idx_j])
                    batch_data_score.append(score)
                    # else:
                    while True:
                        idx_j = np.random.choice(self.avail_vocab, 1)[0]
                        if (idx_j, idx_i) not in self.dev_dict:
                            break 
                    batch_data_pair.append([self.w2embid[idx_j], self.w2embid[idx_i]])
                    score = self.U[idx_j].dot(self.V[idx_i])
                    batch_data_score.append(score)
            yield np.asarray(batch_data_pair), np.asarray(batch_data_score)

