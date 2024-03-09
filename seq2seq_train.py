
import torch
import torch.nn as nn

from seq2seq_data import generate_seq2seq_data
from tokenizer import Tokenizer
from transformer import Seq2SeqTransformer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_padding_mask(src, tgt, PAD_IDX):
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_padding_mask, tgt_padding_mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_attention_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len)
    return src_attention_mask, tgt_attention_mask


if __name__ == '__main__':
    tokenizer = Tokenizer()

    model = Seq2SeqTransformer(src_vocab_size=tokenizer.src_vocab_size, tgt_vocab_size=tokenizer.tgt_vocab_size)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for step in range(1000):
        seq2seq_data = generate_seq2seq_data(batch_size=8, maxlen=10)

        src = tokenizer.tokenize_and_collate([x for x, y in seq2seq_data])[1]
        tgt = tokenizer.tokenize_and_collate([y for x, y in seq2seq_data])[1]

        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_padding_mask, tgt_padding_mask = create_padding_mask(src, tgt_input, tokenizer.PAD_IDX)
        src_attention_mask, tgt_attention_mask = create_attention_mask(src, tgt_input)

        logits = model(src, tgt_input,
                       src_attention_mask=src_attention_mask, tgt_attention_mask=tgt_attention_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(step, loss.item())

    torch.save(model.state_dict(), "model.weights")



