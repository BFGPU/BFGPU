import nltk
import random
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import wordnet
class synonym_replacement:
    def __init__(self, num=5):
        self.num=num
    def augment(self, sentence):
        words = nltk.word_tokenize(sentence)
        new_sentence = sentence
        num=min(self.num,len(words))
        random_index=random.sample(range(len(words)), num)
        for i in range(len(random_index)):
            rand_word_index = random_index[i]
            word=words[rand_word_index]
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
            synonyms = list(synonyms)
            if synonyms:
                synonym = random.choice(synonyms)
                new_sentence = new_sentence.replace(word, synonym)
        return new_sentence

class back_translate:
    def __init__(self,src_lang='en', tgt_lang='fr',device='cuda:0'):
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.device=device
        translate_model_name = f'Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}'
        self.translate_model = MarianMTModel.from_pretrained(translate_model_name).to(device)
        self.translate_tokenizer = MarianTokenizer.from_pretrained(translate_model_name)
        back_translate_model_name = f'Helsinki-NLP/opus-mt-{self.tgt_lang}-{self.src_lang}'
        self.back_translate_model = MarianMTModel.from_pretrained(back_translate_model_name).to(device)
        self.back_translate_tokenizer = MarianTokenizer.from_pretrained(back_translate_model_name)

    def augment(self,sentence):
        device=self.device
        inputs = self.translate_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        translated_ids = self.translate_model.generate(**inputs)
        translated_text = self.translate_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        inputs = self.back_translate_tokenizer(translated_text, return_tensors='pt', padding=True, truncation=True).to(device)
        back_translated_ids = self.back_translate_model.generate(**inputs)
        back_translated_text = self.back_translate_tokenizer.decode(back_translated_ids[0], skip_special_tokens=True)
        return back_translated_text
