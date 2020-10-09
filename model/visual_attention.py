import sys
import os
import random
import logging
import torch
import scipy
import numpy as np
import pandas as pd
from utils.util import oe_score
from utils.util import cosine_distance
from utils.util import asymmetric_distance
from utils.util import load_word_vectors
from model.models import * 
from utils.data_helper_4context import Dataset
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score,precision_recall_curve
import configparser

config = configparser.RawConfigParser()
config.read(sys.argv[1])

from codecs import open
def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
    weights: attention weights for visualizing
    texts: text on which attention weights are to be visualized
    """
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k].split(" ");
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    if (k%2 == 0) {
    var heat_text = "<p><br><b>Example:</b><br>";
    } else {
    var heat_text = "<b>Example:</b><br>";
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\""%x
    #putQuote = lambda x: "%s"%x
    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
  
    return 0


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

def init_model(config, ckpt_path, init_w2v_embedding, device):

    encoder_type = config.get("hyperparameters", "model")
    number_hidden_layers = int(config.getfloat("hyperparameters", "number_hidden_layers"))
    hidden_layer_size = int(config.getfloat("hyperparameters", "hidden_layer_size"))

    model = Context2Score(encoder_type, 300, hidden_layer_size, device)

    pretrain = torch.load(ckpt_path)
    # pretrain.pop("word_embedding.weight")
    model.load_state_dict(pretrain)
    model.init_emb(torch.FloatTensor(init_w2v_embedding))
    model.eval()

    return model

ckpt_dir = config.get("data", "ckpt")
hparam = make_hparam_string(config)
ckpt_dir = os.path.join(ckpt_dir, hparam)
log_path = os.path.join(ckpt_dir, 'eval_last_p.log')
ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')

dataset = Dataset(config, train=False)
gpu_device = config.get("hyperparameters", "gpu_device")
device = torch.device('cuda:{}'.format(gpu_device) if torch.cuda.is_available() else 'cpu')

model = init_model(config, ckpt_path, dataset.context_word_emb, device)

## To visusialize attention 

#hypon = "vicarage"
#hyper = "building"
#hypon = "calamus"
#hyper = "specie"
#hypon = "pontoon"
#hyper = "boat"
#hypon = "polymerase"
#hyper = "enzyme"
#hyper = "chemical"

hypon = "kinetoscope"
hyper = "device"

hypon_id = dataset.context_w2i[hypon]
hyper_id = dataset.context_w2i[hyper]
print(hypon_id)
hypon_word_context = dataset.context_dict[hypon_id]

print(hypon_word_context)

hypon_word = dataset.load_prediction_word_context(hypon_id)
print(hypon_word)


hyper_word_context = dataset.context_dict[hyper_id]
hyper_word = dataset.load_prediction_word_context(hyper_id)
print(hyper_word)

model_name  = config.get("hyperparameters", "model")

inputs = torch.tensor(np.asarray([[hypon_word_context, hyper_word_context]]), dtype=torch.long)
output = model(inputs)
score = output[0].detach().cpu().numpy()[0]

if "han" in model_name:
    attention1 = torch.squeeze(output[1][0]).detach().cpu().numpy()
    h_att1 = torch.squeeze(output[1][1]).detach().cpu().numpy()
    print("The attention weights of hyponymy is : ")
    print(h_att1)

    array = np.asarray(h_att1, dtype=np.float32)
    tmp = array.argsort()

    print(tmp)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(array))

    print(ranks)


    attention2 = torch.squeeze(output[1][2]).detach().cpu().numpy()
    h_att2 = torch.squeeze(output[1][3]).detach().cpu().numpy()
    print("The attention weights of hypernymy is : ")
    print(h_att2)

    text = [hypon_word[tmp[i]][0] + hypon_word[tmp[i]][1] for i in range(len(hypon_word))]

    weights = [attention1[tmp[i]].tolist() for i in range(len(attention1))]

    text2 =  [hyper_word[i][0] + hyper_word[i][1] for i in range(len(hyper_word))]

    for i in range(len(text)):
        for j in range(len(text[0])):
            if '"' or "'" in text[i][j]:
                text[i][j] = text[i][j].replace('"', "/").replace("'","/")

    weights2 = [attention2[i].tolist() for i in range(len(attention2))]
    print(text)
    #print(weights)
    file_name = "vis_" + model_name + "_" + hypon + ".html"
    createHTML(text, weights,  file_name)

else:
    attention1 = torch.squeeze(output[1]).detach().cpu().numpy()
   
    attention2 = torch.squeeze(output[2]).detach().cpu().numpy()
   

    text = [hypon_word[i][0] + hypon_word[i][1][::-1] for i in range(len(hypon_word))]

    print(text)

    att_weights = [attention1[i].tolist() for i in range(len(attention1))]

    print(att_weights)

    weights = [att_weights[i][:10] + att_weights[i][10:][::-1] for i in range(len(att_weights))]

    print(weights)

    text2 =  [hyper_word[i][0] + hyper_word[i][1] for i in range(len(hyper_word))]

    for i in range(len(text)):
        for j in range(len(text[0])):
            if '"' or "'" in text[i][j]:
                text[i][j] = text[i][j].replace('"', "/").replace("'","/")

    weights2 = [attention2[i].tolist() for i in range(len(attention2))]
    print(text)
    #print(weights)
    file_name = "vis_" + model_name + "_" + hypon + ".html"
    createHTML(text, weights,  file_name)



