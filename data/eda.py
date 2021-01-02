import random
from nltk.corpus import wordnet,stopwords

stop_words = list(stopwords.words('english'))
stop_words.extend(['[CLS]','[SEP]','[PAD]'])
random.seed(1)

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n:
			break

	return new_words

def random_deletion(words, p):
    
	if len(words) == 1:
		return words

	new_words = []
	for word in words:
		if word in ['[CLS]','[SEP]','[PAD]']:
    			new_words.append(word)
		else:
			r = random.uniform(0, 1)
			if r > p:
				new_words.append(word)

	return new_words,len(new_words)


def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	while new_words[random_idx_1] in ['[CLS]','[SEP]','[PAD]']:
    		random_idx_1 = random.randint(0, len(new_words)-1)	
	
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1 or new_words[random_idx_2] in ['[CLS]','[SEP]','[PAD]']:
    		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
    			return new_words
	
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words


def random_insertion(words, n,length):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words,length)
	return new_words

def add_word(new_words,length):
	synonyms = []
	while len(synonyms) < 1:
		random_word = new_words[random.randint(1, length-1)]
		synonyms = get_synonyms(random_word)

	random_synonym = synonyms[0]
	random_idx = random.randint(1, length-1)
	new_words.insert(random_idx, random_synonym)

def eda(length,words, mask,sr_rate,rd_rate,sw_rate,ri_rate):
	words_da = [words]
	mask_da = [mask]
	l = list(mask).index(0)

	if(sr_rate > 0):
    		words_da.append(synonym_replacement(words,sr_rate*l))
		mask_da.append(mask)
	
	if(rd_rate > 0):
			new_words,new_length = random_deletion(words,rd_rate)
			delta = length-new_length
			new_words.extend(['[PAD]' for _ in range(delta)])
			new_mask = mask[delta:].extend([0 for _ in range(delta)])
			words_da.append(new_words)
			mask_da.append(new_mask)
	
	if(sw_rate > 0):
    		words_da.append(swap_word(words,sw_rate*l))
		mask_da.append(mask)

	if(ri_rate > 0):
			new_words = random_insertion(words,ri_rate*l,l)[:length]
			if new_words[-1] is not '[PAD]': new_words[-1] = '[SEP]' 
			nl = int(ri_rate*l)+l
			new_mask = [1 for _ in range(nl)].extend([0 for _ in range(length-nl)]) if nl < length else [1 for _ in range(length)]
			words_da.append(new_words)
			mask_da.append(new_mask)
	
	return words_da,mask_da