from datetime import datetime
import stanza
from stanza.models.common.doc import Document

'''
@pre 
    * Install Stanza [https://stanfordnlp.github.io/stanza/]

@brief
    * Finds keywords in responses provided using lemmatization provided by Stanza
    * Uses keywords from [v2_stanza/3_keywords/sample_keywords.txt]
    * Uses responses from [v2_stanza/3_keywords/sample_responses.txt]
'''

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
keywords = open('v2_stanza/3_keywords/sample_keywords.txt', 'r').readlines()
responses = open('v2_stanza/3_keywords/sample_responses.txt', 'r').readlines()

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end = " ")
print("INFO: Finished reading files")


# @brief: checks if word1 + word2 matcches the keyword by lemma
# @input:
#   * keyword is a non-Null string
#   * word1 is a stanza word object or None if there is no first word
#   * word2 is a stanza word object
# @output: True if match and False otherwise
def isMatch(keyword, word1, word2):
    keyWords = keyword.lower().split()
    # Only handles keywords of length 1 or 2
    if(len(keyWords) == 1):
        if(keyWords[0] == word2.lemma):
            return True
        else:
            return False
    elif(len(keyWords) == 2):
        if(word1 == None):
            return False
        elif(keyWords[0] == word1.lemma and keyWords[1] == word2.lemma):
            return True
        else:
            return False
    return False

# @brief: Finds all keywords in response
# @input: resp is a string, keyowrds is a list of strings
# @output: returns all keywordds found in resp as a list
def indivFindKeywords(resp, keywords):
    # Lemmatize words
    repsponseLemmas = nlp(resp).iter_words()
    keywordsFound = []
    prevWord = None
    for word in repsponseLemmas:
        for k_ind in range(len(keywords)):
            if(isMatch(keywords[k_ind], prevWord, word)):
                keywordsFound.append(keywords[k_ind].strip())
        prevWord = word
        
    return keywordsFound

# @brief: Helper function to print keywords
# @input: keys is a list of strings
def printKeywords(keys):
    print("KEYWORDS: ")
    for key in keys:
        print(key + ", ", end= "")
    print()

# @brief: Prints all keywords found in each response
# @input: responses is a list of strings, keywords is a list of strings
def groupFindKeywords(responses, keywords):
    # preprocess keywords
    for resp in responses:
        print("---------------------------------------------")
        keysFound = indivFindKeywords(resp, keywords)
        print("RESPONSE: ")
        print(resp + "\n\n")
        printKeywords(keysFound)
        print("---------------------------------------------")



groupFindKeywords(responses, keywords)


    


