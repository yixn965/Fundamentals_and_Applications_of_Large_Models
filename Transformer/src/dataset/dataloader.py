import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])

def collate_fn(batch,en_vocab,zh_vocab):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch

def get_dataloaders(en_sequences, zh_sequences, batch_size, en_vocab, zh_vocab):

    dataset = TranslationDataset(en_sequences, zh_sequences)

    train_data, val_data = train_test_split(dataset, test_size=0.1)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x,en_vocab,zh_vocab))
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x,en_vocab,zh_vocab))

    return train_dataloader, val_dataloader