import os
from .tokenizer import tokenize
from .config import params
from collections import Counter
import pickle

def save(**kwargs):
	base_dir = params["base_dir"]
	data = kwargs["data"]
	feat = kwargs["feat"]
	fname = "%s.pkl"%(kwargs["feat"])
	fpath = os.path.join(base_dir, fname)
	with open(fpath, "wb+") as f:
		pickle.dump(data, f)
		print("Saving:", kwargs["feat"])

def load(**kwargs):
    print("loading %s"%(kwargs["feat"]))
    base_dir = params["base_dir"]
    fname = "%s.pkl"%(kwargs["feat"])
    fpath = os.path.join(base_dir, fname)

    if os.path.exists(fpath):
    	with open(fpath, 'rb') as fp:
            # pdb.set_trace()
            print(fpath)
            saved = pickle.load(fp)
            return saved
    else:
        return None

def build_data(n_words):
	text_file = os.path.join(params["base_dir"], params["text_file"])
	with open(text_file, 'r') as in_file:
		text = in_file.read()
	words = tokenize(text)
	count = Counter(words).most_common(n_words - 1)
	vocab = {}
	for word, _ in count:
		vocab[word] = len(vocab)
	vocab_inv = {v:k for k,v in vocab.items()}
	def word_to_index():
		return [vocab[word] if word in vocab else 0 for word in words]
	data = {
		"vocabulary": vocab,
		"vocabulary_inverse": vocab_inv,
		"words":word_to_index()
	}
	save(data=data, feat="vocab")
	return data




