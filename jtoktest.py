import transformers

# tokenizer = transformers.BertJapaneseTokenizer.from_pretrained( \
#     'config/bert-base-japanese-vocab.txt' , do_lower_case=False)
tokenizer = transformers.GPT2JapaneseTokenizer.from_pretrained(
    #'config/gpt2-japanese-vocab.txt'
    '/home/knok/nlp/huge-language-models/models/hugface/rebuild-vocab.txt'
)

text = "また、「紙(かみ)」「絵/画(ゑ)」など、もともと音であるが和語と認識されているものもある。"

tokens = tokenizer.tokenize(text)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)