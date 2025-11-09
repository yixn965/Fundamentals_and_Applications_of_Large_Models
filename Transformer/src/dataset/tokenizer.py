import torch
from collections import Counter

class Vocab:
    def __init__(self, counter, specials=None):
        specials = specials or []
        self.itos = specials + [word for word, _ in counter.most_common()]
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def set_default_index(self, index):
        self.default_index = index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

    def __len__(self):
        return len(self.itos)
    
    def lookup_token(self, index):
        return self.itos[index]

def build_vocab_from_iterator(iterator, specials=None):
    counter = Counter()
    
    for sentence in iterator:
        # 如果 sentence 已经是一个列表，直接作为 tokens 使用
        if isinstance(sentence, str):
            tokens = sentence.split()  # 使用空格分词
        else:
            tokens = sentence  # 如果是列表，直接使用
        
        counter.update(tokens)  # 更新词频
    
    return Vocab(counter, specials)

# 构建词汇表函数
def build_vocab(sentences, tokenizer):
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# 定义英文分词器
def get_tokenizer():
    return lambda text: text.split()
tokenizer_en = get_tokenizer()

# 定义中文分词器：将每个汉字作为一个 token
def tokenizer_zh(text):
    return list(text)

def process_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices

