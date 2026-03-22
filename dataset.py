from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

TOKENIZER_NAME = "bert-base-multilingual-cased"
MAX_LEN = 64      
SUBSET_SIZE = 1000  
BATCH_SIZE = 16
PAD_IDX = 0


def load_tokenizer():
    print(f"Carregando tokenizador: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer

def load_translation_data(tokenizer, subset_size=SUBSET_SIZE):
    print("Carregando dataset opus_books (en-pt)...")
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")

    dataset = dataset.select(range(min(subset_size, len(dataset))))
    print(f"Subconjunto carregado: {len(dataset)} pares de frases")

    return dataset

def tokenize_pair(tokenizer, src_text, tgt_text, max_len=MAX_LEN):
    
    src_ids = tokenizer.encode(
        src_text,
        add_special_tokens=True,  
        max_length=max_len,
        truncation=True
    )
 
    tgt_ids = tokenizer.encode(
        tgt_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True
    )

    return src_ids, tgt_ids

class TranslationDataset(Dataset):

    def __init__(self, hf_dataset, tokenizer, max_len=MAX_LEN):
        self.data = []
        print("Tokenizando pares de frases...")

        for item in hf_dataset:
            src_text = item["translation"]["en"]
            tgt_text = item["translation"]["pt"]

            src_ids, tgt_ids = tokenize_pair(tokenizer, src_text, tgt_text, max_len)

            if len(src_ids) < 2 or len(tgt_ids) < 2:
                continue

            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

        print(f"Pares válidos após tokenização: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    
    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return src_padded, tgt_padded

def prepare_dataloader(subset_size=SUBSET_SIZE, batch_size=BATCH_SIZE):
    tokenizer = load_tokenizer()
    hf_dataset = load_translation_data(tokenizer, subset_size)
    dataset = TranslationDataset(hf_dataset, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id or PAD_IDX

    print(f"Vocab size: {vocab_size} | PAD idx: {pad_idx}")
    return loader, vocab_size, pad_idx, tokenizer
