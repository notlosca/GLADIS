import numpy as np
from math import exp
import torch
from torch import nn
from transformers import BertTokenizer, BertForNextSentencePrediction
import utils
from maddog import Extractor
import spacy
import constant

nlp = spacy.load("en_core_web_sm")
ruleExtractor = Extractor()
kb = utils.load_acronym_kb('../input/acronym_kb.json')

class AcronymBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cpu'):
        super().__init__()
        self.device = device
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, sentence):

        samples = self.tokenizer(sentence, padding=True, return_tensors='pt', truncation=True)["input_ids"]
        samples = samples.to(self.device)
        outputs = self.model(samples).logits
        scores = nn.Softmax(dim=1)(outputs)[:, 0]

        return scores


def softmax(elements):
    total = sum([exp(e) for e in elements])
    return exp(elements[0]) / total


def predict(topk, model, short_form, context, batch_size, acronym_kb, device):
    ori_candidate = utils.get_candidate(acronym_kb, short_form)
    long_terms = [str.lower(can) for can in ori_candidate]
    scores = cal_score(model.model, model.tokenizer, long_terms, context, batch_size, device)
    #indexes = [np.argmax(scores)]
    topk = min(len(scores), topk)
    indexes = np.array(scores).argsort()[::-1][:topk]
    names = [ori_candidate[i] for i in indexes]
    return names


def cal_score(model, tokenizer, long_forms, contexts, batch_size, device):
    ps = list()
    for index in range(0, len(long_forms), batch_size):
        batch_lf = long_forms[index:index + batch_size]
        batch_ctx = [contexts] * len(batch_lf)
        encoding = tokenizer(batch_lf, batch_ctx, return_tensors="pt", padding=True, truncation=True, max_length=400).to(device)
        outputs = model(**encoding)
        logits = outputs.logits.cpu().detach().numpy()
        p = [softmax(lg) for lg in logits]
        ps.extend(p)
    return ps


def dog_extract(sentence):
    tokens = [t.text for t in nlp(sentence) if len(t.text.strip()) > 0]
    rulebased_pairs = ruleExtractor.extract(tokens, constant.RULES)
    return rulebased_pairs


def acrobert(sentence, model, device):
    #params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(params)

    tokens = [t.text for t in nlp(sentence) if len(t.text.strip()) > 0]
    print(f"Tokens: {tokens}")
    rulebased_pairs = ruleExtractor.extract(tokens, constant.RULES) # I think this is the source of the problems in detecting lower case
    print(f"Rulebased pairs: {rulebased_pairs}")
    results = list()
    for acronym in rulebased_pairs.keys():
        if rulebased_pairs[acronym][0] != '':
            results.append((acronym, rulebased_pairs[acronym][0]))
        else:

            pred = predict(1, model, acronym, sentence, batch_size=10, acronym_kb=kb, device=device)
            results.append((acronym, pred[0]))
    return results


def popularity(sentence):

    tokens = [t.text for t in nlp(sentence) if len(t.text.strip()) > 0]
    rulebased_pairs = ruleExtractor.extract(tokens, constant.RULES)

    results = list()
    for acronym in rulebased_pairs.keys():
        if rulebased_pairs[acronym][0] != '':
            results.append((acronym, rulebased_pairs[acronym][0]))
        else:

            pred = utils.get_candidate(kb, acronym, can_num=1)
            results.append((acronym, pred[0]))
    return results


def acronym_linker(sentence, model, mode='acrobert', device='cuda:0'):
    if mode == 'acrobert':
        return acrobert(sentence, model, device)
    if mode == 'pop':
        return popularity(sentence)
    raise Exception('mode name should in this list [acrobert, pop]')


if __name__ == '__main__':
    device = 'cuda:0'
    model_path='../input/acrobert.pt'
    model = AcronymBERT(device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    
    text_file_path = '../input/example_text.txt'
    with open(text_file_path, 'r') as f:
        sentences = f.readlines()
    # sentences = [i.strip() for i in sentences]
    full_text = ' '.join(sentences) 
    print(full_text)
    sentences = [full_text]
    # mode = ['acrobert', 'pop']
    for sentence in sentences:
        print(f'Sentence:\t{sentence}')
        results = acronym_linker(sentence, model=model, mode='acrobert')
        print("Results:", results)