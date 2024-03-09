import random
from typing import List, Tuple


def generate_seq2seq_data(batch_size: int, maxlen=10) -> List[Tuple[str, str]]:
    seq2seq_data = []
    src_vocabs = ['a', 'b', 'c', 'A', 'B', 'C']
    for i in range(batch_size):
        src_sentence = ""
        src_len = random.choice(list(range(1, maxlen+1)))
        for j in range(src_len):
            src_sentence += random.choice(src_vocabs)
        tgt_sentence = src_sentence.upper()
        seq2seq_data.append((src_sentence, tgt_sentence))
    return seq2seq_data


if __name__ == '__main__':
    seq2seq_data = generate_seq2seq_data(batch_size=4, maxlen=10)
    print(seq2seq_data)