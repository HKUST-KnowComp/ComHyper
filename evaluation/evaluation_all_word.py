import sys
import time
import os
import logging
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
import configparser
from model.models import *
from model.models import Word2Score
from utils.data_helper import Dataset
from utils.loader import Testdataset
from scipy import stats
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score,precision_recall_curve

SIEGE_EVALUATIONS = [
    ("bless", "data/bless.tsv"),
    ("eval", "data/eval.tsv"),
    ("leds", "data/leds.tsv"),
    ("shwartz", "data/shwartz.tsv"),
    ("weeds", "data/wbless.tsv"),
]

CORRELATION_EVAL_DATASETS = [("hyperlex", "data/hyperlex_rnd.tsv"),
                    ("hyperlex_noun", "data/hyperlex_noun.tsv")]


def predict_many(data, model, hypos, hypers, embedding, device, reverse=False):

    num = 0
    result = []
    result_svd = []
    count_oop = 0
    count_pair = 0
    for hypon, hyper in zip(hypos, hypers):
        count_pair += 1
        if hypon in data.vocab and hyper in data.vocab:
            l = data.word2id[hypon]
            r = data.word2id[hyper]
            
            if reverse:
                pred = data.U[r].dot(data.V[l])
            else:
                pred = data.U[l].dot(data.V[r])
            result_svd.append(pred)

        else: 
        # out of pattern mode
            result_svd.append(0.0)
            count_oop += 1
            if hypon in embedding and hyper in embedding:
                hypon_tensor = torch.from_numpy(embedding[hypon]).view(1,300).to(device)
                hyper_tensor = torch.from_numpy(embedding[hyper]).view(1,300).to(device)

                if reverse:
                    # pred = inference(saved_model,hyper_tensor, hypon_tensor)
                    pred = model.inference(hyper_tensor, hypon_tensor).detach().cpu().numpy()[0]
                else:
                    # pred = inference(saved_model,hypon_tensor, hyper_tensor)
                    pred = model.inference(hypon_tensor, hyper_tensor).detach().cpu().numpy()[0]
            else:
                num +=1
                pred = 0.0

        result.append(pred)
    # num = 0 -> all the word in the embedding
    oop_rate = count_oop * 1.0 / count_pair
    return np.array(result, dtype=np.float32), np.array(result_svd, dtype=np.float32), oop_rate 



def make_hparam_string(config):
    hparam = "{}/s{}_h{}-{}_n{}_w{}".format(
            config.get("hyperparameters", "model"),
            config.get("hyperparameters", "svd_dimension"),
            config.get("hyperparameters", "number_hidden_layers"),
            config.get("hyperparameters", "hidden_layer_size"),
            config.get("hyperparameters", "negative_num"),
            # config.get("hyperparameters", "batch_size"),
            config.get("hyperparameters", "weight_decay"),
            # config.get("hyperparameters", "context_num"),
            # config.get("hyperparameters", "context_len")
            )
    return hparam

def init_model(config):

    hidden_layer_size = int(config.getfloat("hyperparameters", "hidden_layer_size"))
    number_hidden_layers = int(config.getfloat("hyperparameters", "number_hidden_layers"))

    model = Word2Score(hidden_layer_size, number_hidden_layers)
    return model

def load_gensim_word2vec():

    print("Loading pretrained word embedding ... ")
    wv_model = Word2Vec.load("/home/shared/embedding/ukwac.model")
    embedding  = wv_model.wv

    return embedding


def detection_setup(file_name, model, matrix_data, embedding,device):

    ds = Testdataset(file_name, matrix_data.vocab)
    logger.info("-" * 80)
    logger.info("processing dataset :{}".format(file_name))

    m_val = ds.val_mask
    m_test = ds.test_mask

    h = np.zeros(len(ds))
    h_ip = np.zeros(len(ds))

    predict_mask = np.full(len(ds), True)
    inpattern_mask = np.full(len(ds), True)

    true_prediction = []
    in_pattern_prediction = []

    count_w2v = 0

    mask_idx = 0
    for x,y in zip(ds.hypos, ds.hypers):
        if x in matrix_data.vocab and y in matrix_data.vocab:

            l = matrix_data.word2id[x]
            r = matrix_data.word2id[y]
            score = matrix_data.U[l].dot(matrix_data.V[r])

            true_prediction.append(score)
            in_pattern_prediction.append(score)

        else:
            # out of pattern
            inpattern_mask[mask_idx] = False

            if x in embedding and y in embedding:
                hypon_tensor = torch.from_numpy(embedding[x]).view(1,300).to(device)
                hyper_tensor = torch.from_numpy(embedding[y]).view(1,300).to(device)
                score = model.inference(hypon_tensor, hyper_tensor).detach().cpu().numpy()[0]
                true_prediction.append(score)

                count_w2v +=1 

            else:
                predict_mask[mask_idx] = False
        mask_idx +=1 

    h[predict_mask] = np.array(true_prediction, dtype=np.float32)
    h[~predict_mask] = h[predict_mask].min()

    h_ip[inpattern_mask] = np.array(in_pattern_prediction, dtype=np.float32)
    h_ip[~inpattern_mask] = h_ip[inpattern_mask].min()

    y = ds.y    

    result= {
        "ap_val": average_precision_score(y[m_val],h[m_val]),
        "ap_test": average_precision_score(y[m_test],h[m_test]),
    }
    result['oov_rate'] = np.mean(ds.oov_mask)
    result['predict_num'] = int(np.sum(predict_mask))
    result['oov_num'] = int(np.sum(ds.oov_mask))

    logger.info("there are {:2d}/{:2d} pairs appeared in the trained embedding".format(count_w2v, result['oov_num']))
    logger.info("Word2Vec : AP for validation is :{} || for test is :{}".format(result['ap_val'], result['ap_test']))
    logger.info("Svdppmi : AP for validation is :{} || for test is :{}".format(average_precision_score(y[m_val],h_ip[m_val]), 
        average_precision_score(y[m_test],h_ip[m_test]) ))

    return result

def hyperlex_setup(file_name, model, matrix_data, embedding,device):

    logger.info("-" * 80)
    logger.info("processing dataset :{}".format(file_name))

    ds = Testdataset(file_name, matrix_data.vocab, ycolumn='score')

    h = np.zeros(len(ds))

    predict_mask = np.full(len(ds), True)

    true_prediction = []

    mask_idx = 0
    for x,y in zip(ds.hypos, ds.hypers):
        if x in matrix_data.vocab and y in matrix_data.vocab:

            l = matrix_data.word2id[x]
            r = matrix_data.word2id[y]
            score = matrix_data.U[l].dot(matrix_data.V[r])

            true_prediction.append(score)

        else:
            # out of pattern
            if x in embedding and y in embedding:
                hypon_tensor = torch.from_numpy(embedding[x]).view(1,300).to(device)
                hyper_tensor = torch.from_numpy(embedding[y]).view(1,300).to(device)
                score = model.inference(hypon_tensor, hyper_tensor).detach().cpu().numpy()[0]
                true_prediction.append(score)
            else:
                predict_mask[mask_idx] = False

        mask_idx +=1

    h[predict_mask] = np.array(true_prediction, dtype=np.float32)
    h[~predict_mask] = np.median(h[predict_mask])

    y = ds.labels

    m_train = ds.train_mask
    m_val = ds.val_mask
    m_test = ds.test_mask 

    result = {
        "spearman_train": stats.spearmanr(y[m_train], h[m_train])[0],
        "spearman_val": stats.spearmanr(y[m_val], h[m_val])[0],
        "spearman_test": stats.spearmanr(y[m_test], h[m_test])[0],
    }

    result['oov_rate'] = np.mean(ds.oov_mask)
    result['predict_num'] = int(np.sum(predict_mask))
    result['oov_num'] = int(np.sum(ds.oov_mask))

    logger.info("Word2Vec: train cor: {} | test cor:{}".format(result['spearman_train'],result['spearman_test']))

    return result

def dir_bless_setup(model, matrix_data, embedding, device):

    logger.info("-" * 80)
    logger.info("processing dataset : dir_bless") 
    ds = Testdataset("data/bless.tsv", matrix_data.vocab)

    hypos = ds.hypos[ds.y]
    hypers = ds.hypers[ds.y]

    m_val = ds.val_mask[ds.y]
    m_test = ds.test_mask[ds.y]

    h = np.zeros(len(ds))

    pred_score_list = []
    svd_pred_list = []
    count_oop = 0
    count_pair = 0

    for hypon, hyper in zip(hypos, hypers):
        if hypon in matrix_data.vocab and hyper in matrix_data.vocab:
            l = matrix_data.word2id[hypon]
            r = matrix_data.word2id[hyper]

            forward_pred = matrix_data.U[l].dot(matrix_data.V[r])
            reverse_pred = matrix_data.U[r].dot(matrix_data.V[l])

            if forward_pred > reverse_pred:
                pred_score_list.append(1)
                svd_pred_list.append(1)
            else:
                pred_score_list.append(0)
                svd_pred_list.append(0)
        else: 
        # out of pattern mode
            svd_pred_list.append(0)
            count_oop += 1

            if hypon in embedding and hyper in embedding:
                hypon_tensor = torch.from_numpy(embedding[hypon]).view(1,300).to(device)
                hyper_tensor = torch.from_numpy(embedding[hyper]).view(1,300).to(device)
                forward_pred = model.inference(hypon_tensor, hyper_tensor).detach().cpu().numpy()[0]
                reverse_pred = model.inference(hyper_tensor, hypon_tensor).detach().cpu().numpy()[0]

                if forward_pred > reverse_pred:
                    pred_score_list.append(1)
                else:
                    pred_score_list.append(0)
            else:
                pred_score_list.append(0)       
    
    acc = np.mean(np.asarray(pred_score_list))
    acc_val = np.mean(np.asarray(pred_score_list)[m_val])
    acc_test = np.mean(np.asarray(pred_score_list)[m_test])

    s_acc = np.mean(np.asarray(svd_pred_list))  

    logger.info("Val Acc : {} ||  Test Acc: {} ".format(acc_val, acc_test))
    logger.info("Sppmi Acc: {} ".format(s_acc))


def dir_wbless_setup(model, data, embedding,device):
    
    logger.info("-" * 80)
    logger.info("processing dataset : dir_wbless") 
    data_path = "data/wbless.tsv" 
    ds = Testdataset(data_path, data.vocab)

    rng = np.random.RandomState(42)
    VAL_PROB = .02
    NUM_TRIALS = 1000

        # We have no way of handling oov
    h, h_svd, _ = predict_many(data, model,  ds.hypos, ds.hypers, embedding, device)
    y = ds.y

    val_scores = []
    test_scores = []

    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
            # Test is everything except val
        m_test = ~m_val
        _, _, t = precision_recall_curve(y[m_val], h[m_val])
            # pick the highest accuracy on the validation set
        thr_accs = np.mean((h[m_val, np.newaxis] >= t) == y[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]
        preds_val = h[m_val] >= best_t
        preds_test = h[m_test] >= best_t
            # Evaluate
        val_scores.append(np.mean(preds_val == y[m_val]))
        test_scores.append(np.mean(preds_test == y[m_test]))
            # sanity check
        assert np.allclose(val_scores[-1], thr_accs.max())

        # report average across many folds
    logger.info("w2v: acc_val_inv: {} acc_test_inv: {}".format(np.mean(val_scores), np.mean(test_scores)))

    val_scores = []
    test_scores = []

    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
            # Test is everything except val
        m_test = ~m_val
        _, _, t = precision_recall_curve(y[m_val], h_svd[m_val])
            # pick the highest accuracy on the validation set
        thr_accs = np.mean((h_svd[m_val, np.newaxis] >= t) == y[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]
        preds_val = h_svd[m_val] >= best_t
        preds_test = h_svd[m_test] >= best_t
            # Evaluate
        val_scores.append(np.mean(preds_val == y[m_val]))
        test_scores.append(np.mean(preds_test == y[m_test]))
            # sanity check
        assert np.allclose(val_scores[-1], thr_accs.max())

        # report average across many folds
    logger.info("sppmi: acc_val_inv: {} acc_test_inv: {}".format(np.mean(val_scores), np.mean(test_scores)))    


def dir_bibless_setup(model, data, embedding, device):
    
    logger.info("-" * 80)
    logger.info("processing dataset : dir_bibless") 
    data_path = "data/bibless.tsv" 
    ds = Testdataset(data_path, data.vocab)

    
    rng = np.random.RandomState(42)
    VAL_PROB = .02
    NUM_TRIALS = 1000


     #y = ds.y[ds.invocab_mask]
    y = ds.y
    # hypernymy could be either direction
    yh = y != 0

            # get forward and backward predictions
    hf, hf_svd, oop_rate = predict_many(data, model, ds.hypos, ds.hypers, embedding, device, reverse=False)
    hr, hr_svd, _ = predict_many(data, model, ds.hypos, ds.hypers, embedding, device, reverse=True)
    logger.info('OOP Rate: {}'.format(oop_rate))
    h = np.max([hf, hr], axis=0)
    h_svd = np.max([hf_svd, hr_svd], axis=0)

    dir_pred = 2 * np.float32(hf >= hr) - 1
    dir_pred_svd = 2 * np.float32(hf_svd >= hr_svd) - 1

    val_scores = []
    test_scores = []
    for _ in range(NUM_TRIALS):
                # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
        # Test is everything except val
        m_test = ~m_val

        # set the threshold based on the maximum score
        _, _, t = precision_recall_curve(yh[m_val], h[m_val])
        thr_accs = np.mean((h[m_val, np.newaxis] >= t) == yh[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]

        det_preds_val = h[m_val] >= best_t
        det_preds_test = h[m_test] >= best_t

        fin_preds_val = det_preds_val * dir_pred[m_val]
        fin_preds_test = det_preds_test * dir_pred[m_test]

        val_scores.append(np.mean(fin_preds_val == y[m_val]))
        test_scores.append(np.mean(fin_preds_test == y[m_test]))

                # report average across many folds
    logger.info("w2v: acc_val_all: {}, acc_test_all: {}".format(np.mean(val_scores),np.mean(test_scores)))

    val_scores = []
    test_scores = []
    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        m_val = rng.rand(len(y)) < VAL_PROB
        # Test is everything except val
        m_test = ~m_val

        # set the threshold based on the maximum score
        _, _, t = precision_recall_curve(yh[m_val], h_svd[m_val])
        thr_accs = np.mean((h_svd[m_val, np.newaxis] >= t) == yh[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]

        det_preds_val = h_svd[m_val] >= best_t
        det_preds_test = h_svd[m_test] >= best_t

        fin_preds_val = det_preds_val * dir_pred_svd[m_val]
        fin_preds_test = det_preds_test * dir_pred_svd[m_test]

        val_scores.append(np.mean(fin_preds_val == y[m_val]))
        test_scores.append(np.mean(fin_preds_test == y[m_test]))

                # report average across many folds
    logger.info("sppmi: acc_val_all: {}, acc_test_all: {}".format(np.mean(val_scores),np.mean(test_scores)))



def evaluation_all(model_config):

    embedding = load_gensim_word2vec()
    config = configparser.RawConfigParser()

    config.read(model_config)

    gpu_device = config.get("hyperparameters", "gpu_device")
    device = torch.device('cuda:{}'.format(gpu_device) if torch.cuda.is_available() else 'cpu')

    matrix_data = Dataset(config)

    model = init_model(config)
    model.to(device)

    #pretrain = torch.load("/home/shared/acl-data/hype_detection/checkpoints/mlp_unisample_svd/s50_h2-300_n400_w0/best.ckpt")
    pretrain = torch.load("/home/cyuaq/comHyper/checkpoints/mlp_unisample_svd/s50_h2-300_n400_b128/best.ckpt")
    pretrain.pop("embs.weight")
    model.load_state_dict(pretrain)
    model.eval()
    
    results = {}

    for taskname, filename in SIEGE_EVALUATIONS:
        result = detection_setup(filename, model, matrix_data, embedding,device)
        results["detec_{}".format(taskname)] = result

    for taskname, filename in CORRELATION_EVAL_DATASETS:
        result = hyperlex_setup(filename, model, matrix_data, embedding, device)
        results["corr_{}".format(taskname)] = result

    dir_bless_setup(model, matrix_data, embedding, device)
    dir_wbless_setup(model, matrix_data, embedding, device)
    dir_bibless_setup(model, matrix_data, embedding, device)

    return results

if __name__ == "__main__":

    config_file = sys.argv[1]

    log_path = "/home/cyuaq/comHyper/checkpoints/mlp_unisample_svd/s50_h2-300_n400_b128/word2score.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    results = evaluation_all(config_file)
    print(results)
