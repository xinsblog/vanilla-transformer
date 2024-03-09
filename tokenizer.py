from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:

    def __init__(self):
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.token2id = {
            '<unk>': self.UNK_IDX,
            '<pad>': self.PAD_IDX,
            '<bos>': self.BOS_IDX,
            '<eos>': self.EOS_IDX,
            'a': 4,
            'b': 5,
            'c': 6,
            'A': 7,
            'B': 8,
            'C': 9
        }
        self.id2token = {
            v: k for k, v in self.token2id.items()
        }
        self.src_vocab_size = len(self.token2id)
        self.tgt_vocab_size = len(self.token2id)

    def tokenize_and_collate(self, texts: List[str]):
        batch_tokens, batch_token_ids = [], []
        for text in texts:
            tokens, token_ids = self.tokenize(text)
            batch_tokens.append(tokens)
            batch_token_ids.append(token_ids)
        batch_token_ids = pad_sequence(batch_token_ids, padding_value=self.PAD_IDX)
        return batch_tokens, batch_token_ids

    def tokenize(self, text: str):
        tokens = ['<bos>']
        token_ids = [self.BOS_IDX]
        for c in text:
            if c in self.token2id:
                tokens.append(c)
                token_ids.append(self.token2id[c])
            else:
                tokens.append('<unk>')
                token_ids.append(self.UNK_IDX)
        tokens.append('<eos>')
        token_ids.append(self.EOS_IDX)
        token_ids = torch.LongTensor(token_ids)
        return tokens, token_ids


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.tokenize('ab123ab'))
    print(tokenizer.tokenize_and_collate(['a', 'ABC', 'ab123ab']))
