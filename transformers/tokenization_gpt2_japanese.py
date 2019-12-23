# coding=utf-8
# Copyright 2018-2019 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Japanese Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import collections
import logging
import os
import regex as re
from io import open

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer, load_vocab
from .tokenization_bert_japanese import MecabTokenizer, CharacterTokenizer
from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'gpt2-japanese': "https://s3.amazonaws.com/models.huggingface.co/bert/knok/gpt2-vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'gpt2-japanese': 1024,
}

PRETRAINED_INIT_CONFIGURATION = {
    'gpt2-japanese': {
        'do_lower_case': False,
        'word_tokenizer_type': 'mecab',
        'subword_tokenizer_type': 'wordpiece'
    },
}

class GPT2JapaneseTokenizer(PreTrainedTokenizer):
    """GPT-2 Japanese Tokenizer"""
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=False,
                 do_word_tokenize=True, do_subword_tokenize=True,
                 word_tokenizer_type='basic', subword_tokenizer_type='wordpiece', never_split=None,
                 errors='replace', unk_token="[UNK]",
                 bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):
        super(GPT2JapaneseTokenizer, self).__init__(vocab_file=vocab_file,bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        if self.max_len > 1e4:
            self.max_len = 1024 # shrink
        self.max_len_single_sentence = self.max_len # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = self.max_len # no default special tokens - you can update this value if you add special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_word_tokenize = do_word_tokenize
        if do_word_tokenize:
            if word_tokenizer_type == 'basic':
                self.word_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                     never_split=never_split,
                                                     tokenize_chinese_chars=False)
            elif word_tokenizer_type == 'mecab':
                self.word_tokenizer = MecabTokenizer(do_lower_case=do_lower_case,
                                                     never_split=never_split)
            else:
                raise ValueError(
                    "Invalid word_tokenizer_type '{}' is specified.".format(word_tokenizer_type))

        self.do_subword_tokenize = do_subword_tokenize
        if do_subword_tokenize:
            if subword_tokenizer_type == 'wordpiece':
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                            unk_token=self.unk_token)
            elif subword_tokenizer_type == 'character':
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab,
                                                            unk_token=self.unk_token)
            else:
                raise ValueError(
                    "Invalid subword_tokenizer_type '{}' is specified.".format(subword_tokenizer_type))

    @property
    def vocab_size(self):
        return len(vocab)

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text,
                                                  never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens
                            for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
