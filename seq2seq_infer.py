
import torch

from tokenizer import Tokenizer
from transformer import Seq2SeqTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, tokenizer: Tokenizer, src_sentence: str):
    model.eval()

    src = tokenizer.tokenize(src_sentence)[1].view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5,
        start_symbol=tokenizer.BOS_IDX, end_symbol=tokenizer.EOS_IDX).flatten()
    tgt_sentence = "".join([tokenizer.id2token[token_id] for token_id in tgt_tokens.cpu().numpy()])
    tgt_sentence = tgt_sentence.replace("<bos>", "").replace("<eos>", "")
    print(tgt_sentence)


if __name__ == '__main__':
    tokenizer = Tokenizer()

    model = Seq2SeqTransformer(src_vocab_size=tokenizer.src_vocab_size, tgt_vocab_size=tokenizer.tgt_vocab_size)
    model.load_state_dict(torch.load('model.weights', map_location=DEVICE))

    translate(model, tokenizer, src_sentence="a")
    translate(model, tokenizer, src_sentence="aA")
    translate(model, tokenizer, src_sentence="abcABC")





