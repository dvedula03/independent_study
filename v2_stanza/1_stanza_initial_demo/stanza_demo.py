from html import entities
import stanza
from stanza.models.common.doc import Document
# nlp = stanza.Pipeline(lang='zh', processors='tokenize')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
Stanza_doc_open = open('v2_stanza/1_stanza_initial_demo/input.txt', 'r').read()

doc = nlp(Stanza_doc_open)

# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
# doc = nlp('Barack Obama was born in Hawaii.')
# print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')


# nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=True)
# pp = Document([[{'id': 1, 'text': 'puppies', 'upos': 'NOUN'}]])
# print(pp)
# doc = nlp(pp)
print("AFTER ADDING LEMMA")
print(doc.entities)
# print(entities(doc))
# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
# doc = nlp('My name is Bob Sherwood')
# print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')