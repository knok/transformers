import argparse
from pathlib import Path

import numpy as np
import tqdm
from transformers import GPT2JapaneseTokenizer #, BertJapaneseTokenizer

tokenizer = GPT2JapaneseTokenizer("/home/knok/nlp/huge-language-models/models/hugface/gpt2-japanese-vocab.txt", unk_token="[UNK]")
# tokenizer = BertJapaneseTokenizer("/home/knok/nlp/huge-language-models/models/hugface/bert-base-japanese-vocab.txt")

text = "大学からの緊急避難:そのとき、台湾当局はどう動いたのか"
tokens = tokenizer.tokenize(text)
tokenized = tokenizer.convert_tokens_to_ids(tokens)
# print(tokenized, tokens)
example = tokenizer.build_inputs_with_special_tokens(tokenized)
print(example)
print(tokenized)