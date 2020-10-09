import numpy as np
import gensim 
import time
from gensim.models import Word2Vec
from scipy.linalg import norm


class MySentences(object):
	def __init__(self, fileName):
		self.fileName = fileName

	def __iter__(self):
		for line in open(self.fileName, "r"):
			yield line.split()


def cosine_distance(v1, v2):

	return np.dot(v1,v2) / (norm(v1) * norm(v2))


def oe_score(hypo, hyper):

	sub = np.subtract(hypo, hyper)

	mid = np.maximum(sub,0)
	pw = norm(mid, 2)

	norm1 = norm(hypo, 2)
	norm2 = norm(hyper, 2)
	norm_sum = norm1 + norm2

	return float(pw)/norm_sum


def asymmetric_distance(v1, v2, distance_metric):
    """
    Directly copy from LEAR code
    """
    #return distance(v1, v2) + norm(v1) - norm(v2) 
    
    cosine_similarity = cosine_distance(v1,v2)

    norm1 = norm(v1, ord=2)
    norm2 = norm(v2, ord=2)

    if distance_metric == "metric_1":
        # |x| - |y|
        return cosine_similarity+ (norm2 - norm1)

    elif distance_metric == "metric_2":
        # (|x| - |y|) / (|x| + |y|)
        
        norm_difference = norm2 - norm1
        norm_sum = norm1 + norm2

        return cosine_similarity + (norm_difference / norm_sum)

    elif distance_metric == "metric_3":

        max_norm = np.maximum(norm1, norm2)
        norm_difference = norm2 - norm1

        return cosine_similarity + (norm_difference / max_norm)



def load_word_vectors(file_path):

	print("loading vectors from ", file_path)
	input_dict = {}

	with open(file_path, "r") as in_file:
		lines = in_file.readlines()

	in_file.close()

	for line in lines:
		item = line.strip().split()
		dkey = item.pop(0)
		if len(item)!=300:
			continue
		vectors = np.array(item, dtype='float32')
		input_dict[dkey] = vectors

	print(len(input_dict), "vectors load from", file_path)

	return input_dict

def train_word2vec(file_path, saved_path):

	st = time.time()
	sentences = MySentences(file_path) # a memory-friendly iterator
	model = gensim.models.Word2Vec(sentences, size=300, min_count=1, workers=30, iter=5)
	model.save(saved_path + "/unwak.model")
	print('Finished in {:.2f}'.format(time.time()-st))


def load_gensim_word2vec(model_name, word_list, saved_path=None):

	print("Start to load word embedding ...")
	
	model = Word2Vec.load(model_name)
	emb = model.wv
	#emb = load_word_vectors("/home/cyuaq/embeddings/glove.840B.300d.txt")

	input_dict = {}

	#out = open(saved_path, "w")
	num = 0
	for word in word_list:
		if word in emb:
			num +=1 
			input_dict[word] = emb[word]
			#vec = " ".join([str(each) for each in emb[word]])
			#out.write(word + " " + vec + "\n")
	#out.close()
	print("There are total word in word2vec: ",num)
	return input_dict


def load_phrase_word2vec(model_name, word_list, saved_path=None):

    print("Start to load word embeddings ... ")
    model = Word2Vec.load(model_name)

    emb = model.wv
    input_dict = {}

    num = 0 
    for word in word_list:
        if word in emb:
            num+=1
            input_dict[word] = emb[word]
        else:
            if '_' in word:
                tmp = word.split("_")
                tmp_vec = np.zeros(300,dtype=np.float32)
                flag = True
                for w in tmp:
                    if w not in emb:
                        flag = False
                        break
                    else:
                        tmp_vec += emb[w]
                if flag:
                    num +=1
                    input_dict[word] = tmp_vec/len(tmp)

    print("There are total word in word2vec: ",num)
    assert num == len(input_dict)
    return input_dict                