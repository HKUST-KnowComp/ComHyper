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
from model.models import OovRegression
from utils.data_helper_4bert import Dataset
from utils.loader import Testdataset
from scipy import stats
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score,precision_recall_curve
from collections import OrderedDict

SIEGE_EVALUATIONS = [
    ("bless", "data/bless.tsv"),
    ("eval", "data/eval.tsv"),
    ("leds", "data/leds.tsv"),
    ("shwartz", "data/shwartz.tsv"),
    ("weeds", "data/wbless.tsv"),
]

CORRELATION_EVAL_DATASETS = [("hyperlex", "data/hyperlex_rnd.tsv"),
                    ("hyperlex_noun", "data/hyperlex_noun.tsv")]


def make_hparam_string(config):
    hparam = "{}/s{}_h{}-{}_n{}_c{}-{}_b{}".format(
            config.get("hyperparameters", "model"),
            config.get("hyperparameters", "svd_dimension"),
            config.get("hyperparameters", "number_hidden_layers"),
            config.get("hyperparameters", "hidden_layer_size"),
            config.get("hyperparameters", "negative_num"),
            # config.get("hyperparameters", "weight_decay"),
            config.get("hyperparameters", "context_num"),
            config.get("hyperparameters", "context_len"),
            config.get("hyperparameters", "batch_size")
            )
    return hparam

def init_model(config, ckpt_path, device):

    encoder_type = config.get("hyperparameters", "model")
    number_hidden_layers = int(config.getfloat("hyperparameters", "number_hidden_layers"))
    hidden_layer_size = int(config.getfloat("hyperparameters", "hidden_layer_size"))
    bert_dir = config.get("data", "bert_path")
    model = Bert2Score(encoder_type, bert_dir, hidden_layer_size, 0.1)
    model.to(device)
    pretrain = torch.load(ckpt_path)
    new_pretrain = OrderedDict()
    #for k, v in pretrain.items():
    #    name = k[7:]
    #    new_pretrain[name] = v
    # pretrain.pop("word_embedding.weight")
    model.load_state_dict(pretrain)
    model.eval()

    return model


def predict_many(data, model, hypos, hypers, reverse, device):

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
            try:
                hypon_id = data.context_w2i[hypon]
                hyper_id = data.context_w2i[hyper]
                hypon_word_context = data.context_dict[hypon_id]['ids']
                hyper_word_context = data.context_dict[hyper_id]['ids']
      
                hypon_word_mask = data.context_dict[hypon_id]['mask']
                hyper_word_mask = data.context_dict[hyper_id]['mask']

              
                if reverse:
                    inputs = torch.tensor(np.asarray([[hyper_word_context, hypon_word_context]]), dtype=torch.long).to(device)
                    inputs_mask = torch.tensor(np.asarray([[hyper_word_mask, hypon_word_mask]]), dtype=torch.long).to(device)
                    pred = model(inputs, inputs_mask).detach().cpu().numpy()[0]
                else:
                    inputs = torch.tensor(np.asarray([[hypon_word_context, hyper_word_context]]), dtype=torch.long).to(device)
                    inputs_mask = torch.tensor(np.asarray([[hypon_word_mask, hyper_word_mask]]), dtype=torch.long).to(device)
                    pred = model(inputs, inputs_mask).detach().cpu().numpy()[0]
            except:
                num +=1
                pred = 0.0

        result.append(pred)
    # num = 0 -> all the word in the embedding
    oop_rate = count_oop * 1.0 / count_pair
    return np.array(result, dtype=np.float32), np.array(result_svd, dtype=np.float32), oop_rate 


def detection_setup(file_name, model, matrix_data ,device):

    logger.info("-" * 80)
    logger.info("processing dataset :{}".format(file_name))
    ds = Testdataset(file_name, matrix_data.vocab)

    m_val = ds.val_mask
    m_test = ds.test_mask

    h = np.zeros(len(ds))

    h_ip = np.zeros(len(ds))

    print(len(ds))

    predict_mask = np.full(len(ds), True)
    inpattern_mask = np.full(len(ds), True)

    true_prediction = []
    in_pattern_prediction = []

    count_context = 0

    mask_idx = 0
    for x,y in zip(ds.hypos, ds.hypers):
        if x in matrix_data.vocab and y in matrix_data.vocab:

            l = matrix_data.word2id[x]
            r = matrix_data.word2id[y]
            score = matrix_data.U[l].dot(matrix_data.V[r])

            in_pattern_prediction.append(score)
            true_prediction.append(score)
        else:

            inpattern_mask[mask_idx] = False
            # out of pattern
            try:
                hypon_id = matrix_data.context_w2i[x]
                hyper_id = matrix_data.context_w2i[y]
                
                hypon_word_context = matrix_data.context_dict[hypon_id]['ids']
                hyper_word_context = matrix_data.context_dict[hyper_id]['ids']

                hypon_word_mask = matrix_data.context_dict[hypon_id]['mask']
                hyper_word_mask = matrix_data.context_dict[hyper_id]['mask']
                
                inputs = torch.tensor(np.asarray([[hypon_word_context, hyper_word_context]]), dtype=torch.long).to(device)
                inputs_mask = torch.tensor(np.asarray([[hypon_word_mask, hyper_word_mask]]), dtype=torch.long).to(device)
                score = model(inputs, inputs_mask).detach().cpu().numpy()[0]
                count_context +=1

                true_prediction.append(score)

            except Exception as e:
                print(repr(e))
                print(file_name)
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

    result['true_oov'] = int(np.sum(ds.oov_mask & ds.y))

    result['oov_rate'] = np.mean(ds.oov_mask)
    result['predict_num'] = int(np.sum(predict_mask))
    result['oov_num'] = int(np.sum(ds.oov_mask))

    logger.info("there are {:2d}/{:2d} pairs have context".format(count_context, result['oov_num']))
    logger.info("Bert : AP for validation is :{} || for test is :{}".format(average_precision_score(y[m_val],h[m_val]), 
        average_precision_score(y[m_test],h[m_test]) ))
    logger.info("Svdppmi : AP for validation is :{} || for test is :{}".format(average_precision_score(y[m_val],h_ip[m_val]), 
        average_precision_score(y[m_test],h_ip[m_test]) ))
    logger.info("OOV true number is ".format(result['true_oov']))

    return result

def hyperlex_setup(file_name, model, matrix_data,device):

    logger.info("-" * 80)
    logger.info("processing dataset :{}".format(file_name))

    ds = Testdataset(file_name, matrix_data.vocab, ycolumn='score')

    h = np.zeros(len(ds))
    h_ip = np.zeros(len(ds))

    predict_mask = np.full(len(ds), True)
    inpattern_mask = np.full(len(ds), True)

    true_prediction = []
    in_pattern_prediction = []

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
            try:
                hypon_id = matrix_data.context_w2i[x]
                hyper_id = matrix_data.context_w2i[y]
                
                hypon_word_context = matrix_data.context_dict[hypon_id]['ids']
                hyper_word_context = matrix_data.context_dict[hyper_id]['ids']

                hypon_word_mask = matrix_data.context_dict[hypon_id]['mask']
                hyper_word_mask = matrix_data.context_dict[hyper_id]['mask']
                
                inputs = torch.tensor(np.asarray([[hypon_word_context, hyper_word_context]]), dtype=torch.long).to(device)
                inputs_mask = torch.tensor(np.asarray([[hypon_word_mask, hyper_word_mask]]), dtype=torch.long).to(device)
                
                score = model(inputs, inputs_mask).detach().cpu().numpy()[0]

                true_prediction.append(score)

            except Exception as e:
                print(repr(e))
                print(file_name)
                predict_mask[mask_idx] = False
            
        mask_idx +=1

    h[predict_mask] = np.array(true_prediction, dtype=np.float32)
    h[~predict_mask] = np.median(h[predict_mask])

    h_ip[inpattern_mask] = np.array(in_pattern_prediction, dtype=np.float32)
    h_ip[~inpattern_mask] = np.median(h_ip[inpattern_mask])


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

    svd_train = stats.spearmanr(y[m_train], h_ip[m_train])[0]
    svd_test = stats.spearmanr(y[m_test], h_ip[m_test])[0]

    oov_train = stats.spearmanr(y[ds.oov_mask], h[ds.oov_mask])

    logger.info("Bert: train cor: {} | test cor:{}".format(result['spearman_train'],result['spearman_test']))
    logger.info("OOV cor: {}".format(oov_train))

    logger.info("Svdppmi: train cor: {} | test cor:{}".format(svd_train, svd_test))

    return result


def dir_bless_setup(model, matrix_data ,device):

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
            try:
                hypon_id = matrix_data.context_w2i[hypon]
                hyper_id = matrix_data.context_w2i[hyper]
                hypon_word_context = matrix_data.context_dict[hypon_id]['ids']
                hyper_word_context = matrix_data.context_dict[hyper_id]['ids']
                
                hypon_word_mask = matrix_data.context_dict[hypon_id]['ids']
                hyper_word_mask = matrix_data.context_dict[hyper_id]['ids']
                inputs = torch.tensor(np.asarray([[hypon_word_context, hyper_word_context]]), dtype=torch.long).to(device)
                inputs_mask = torch.tensor(np.asarray([[hypon_word_mask, hyper_word_mask]]), dtype=torch.long).to(device)
                
                forward_pred = model(inputs, inputs_mask).detach().cpu().numpy()[0]
                
                inputs = torch.tensor(np.asarray([[hyper_word_context, hypon_word_context]]), dtype=torch.long).to(device)
                inputs_mask = torch.tensor(np.asarray([[hyper_word_mask, hypon_word_mask]]), dtype=torch.long).to(device)

                reverse_pred = model(inputs, inputs_mask).detach().cpu().numpy()[0]

                if forward_pred > reverse_pred:
                    pred_score_list.append(1)
                else:
                    pred_score_list.append(0)
            except Exception as e:
                print(repr(e))
                pred_score_list.append(0) 
    
    acc = np.mean(np.asarray(pred_score_list))
    acc_val = np.mean(np.asarray(pred_score_list)[m_val])
    acc_test = np.mean(np.asarray(pred_score_list)[m_test])

    s_acc = np.mean(np.asarray(svd_pred_list))  

    logger.info("Val Acc : {} ||  Test Acc: {} ".format(acc_val, acc_test))
    logger.info("Sppmi Acc: {} ".format(s_acc))


def dir_wbless_setup(model, data ,device):

    logger.info("-" * 80)
    logger.info("processing dataset : dir_wbless") 
    data_path = "data/wbless.tsv" 
    ds = Testdataset(data_path, data.vocab)


    rng = np.random.RandomState(42)
    VAL_PROB = .02
    NUM_TRIALS = 1000

        # We have no way of handling oov
    h, h_svd, _ = predict_many(data, model, ds.hypos, ds.hypers, False, device)
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
    logger.info("bert: acc_val_inv: {} acc_test_inv: {}".format(np.mean(val_scores), np.mean(test_scores)))

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


def dir_bibless_setup(model, data ,device):

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
    hf, hf_svd, oop_rate = predict_many(data, model, ds.hypos, ds.hypers, False, device)
    hr, hr_svd, _ = predict_many(data, model, ds.hypos, ds.hypers, True, device)
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
    logger.info("bert: acc_val_all: {}, acc_test_all: {}".format(np.mean(val_scores),np.mean(test_scores)))

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

def evaluation_all(config, ckpt_path):

    #embedding = load_gensim_word2vec()
    
    matrix_data = Dataset(config,train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = init_model(config, ckpt_path, device)
    
    results = {}

    for taskname, filename in SIEGE_EVALUATIONS:
        result = detection_setup(filename, model, matrix_data ,device)
        results["detec_{}".format(taskname)] = result

    for taskname, filename in CORRELATION_EVAL_DATASETS:
        result = hyperlex_setup(filename, model, matrix_data, device)
        results["corr_{}".format(taskname)] = result

    dir_bless_setup(model,matrix_data, device)
    dir_wbless_setup(model, matrix_data, device)
    dir_bibless_setup(model, matrix_data, device)

    return results

if __name__ == "__main__":

    config_file = sys.argv[1]
    config = configparser.RawConfigParser()
    config.read(config_file)

    ckpt_dir = config.get("data", "ckpt")
    hparam = make_hparam_string(config)
    ckpt_dir = os.path.join(ckpt_dir, hparam)
    ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')
    log_path = os.path.join(ckpt_dir, 'bert_res.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    results = evaluation_all(config, ckpt_path)
    print(results)
