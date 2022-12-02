from datetime import datetime
import stanza
from stanza.models.common.doc import Document

'''
@pre 
    * Install Stanza [https://stanfordnlp.github.io/stanza/]

@brief
    * Matches keywords provided by mentors to menteer' responses and returns a mapping of mentor 
      index to the mentee indices that use these keywords' lemmas using Stanza. The active sheet 
      ensures that the mentor and mentees ccompared are active (1) and not inactive (0).
    * Uses mentees from [v2_stanza/2_matching_mentors_to_mentees/mentees_responses.txt]
    * Uses mentors from [v2_stanza/2_matching_mentors_to_mentees/mentors_active.txt]
    * Uses mentees active from [v2_stanza/2_matching_mentors_to_mentees/mentees_active.txt]
    * Uses mentors active from [v2_stanza/2_matching_mentors_to_mentees/mentors_active.txt]
'''

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
mentees_file = open('v2_stanza/2_matching_mentors_to_mentees/mentees_responses.txt', 'r').readlines()
mentors_file = open('v2_stanza/2_matching_mentors_to_mentees/mentors_responses.txt', 'r').readlines()
mentees_active = open('v2_stanza/2_matching_mentors_to_mentees/mentees_active.txt', 'r').readlines()
mentors_active = open('v2_stanza/2_matching_mentors_to_mentees/mentors_active.txt', 'r').readlines()

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end = " ")
print("INFO: Finished reading files")

def prettyPrintSetLemmas(words):
    count = 0
    for word in words:
        print(word.lemma, end = ", ")
        count += 1
        if(count % 15 == 0 and count != 0):
            print()

def prettyPrintSortedWords(sortedWords, rNum):
    print("\n---------------------------------------")
    print("Response " + str(rNum) + " Analysis:")
    print("\nNouns:\n")
    prettyPrintSetLemmas(sortedWords["nouns"])
    print()

    print("\nAdjectives:\n")
    prettyPrintSetLemmas(sortedWords["adjectives"])
    print()

    print("\nVerbs:\n")
    prettyPrintSetLemmas(sortedWords["verbs"])
    print()

# Separates words into nouns, adjectives, 
def indivResponseAnalysisSets(resp):
    # Lemmatize words
    lemm = nlp(resp).iter_words()

    nouns = set()
    adj = set()
    verbs = set()
    propNouns = set()
    other = set()

    for w in lemm:
        if(w.upos == "ADJ"):
            adj.add(w)
        elif(w.upos == "NOUN"):
            nouns.add(w)
        elif(w.upos == "PROPN"):
            propNouns.add(w)
        elif(w.upos == "VERB"):
            verbs.add(w)
        else:
            other.add(w)
    return dict({"nouns" : nouns, "proper nouns" : propNouns, "verbs" : verbs, "adjectives" : adj, "other" : other})

def groupAnalysis(active, lines):
    analysis = dict()
    # Individual Responses Analysis
    for rnum in range(len(active)):
        sortedWords = indivResponseAnalysisSets(lines[rnum])
        analysis[rnum] = [int(active[rnum]), sortedWords]
        # prettyPrintSortedWords(sortedWords, rnum)

    return analysis
    
def overlapSets(givenFirst, givenSecond):
    first = set(map((lambda a : a.lemma), givenFirst))
    second = set(map((lambda a : a.lemma), givenSecond))
    for word in first:
        if(word in second):
            return True
    return False

def overlapPeople(p1SortedWords, p2SortedWords):
    return overlapSets(p1SortedWords["nouns"], p2SortedWords["nouns"]) or overlapSets(p1SortedWords["verbs"], p2SortedWords["verbs"]) or overlapSets(p1SortedWords["adjectives"], p2SortedWords["adjectives"])

def matches(tors, tees):
    activeCompatibles = dict()
    for tee in tees:
        if(tees[tee][0]):
            matched = []
            for tor in tors:
                if(tors[tor][0]):
                    # both active, check if they are a match
                    if(overlapPeople(tees[tee][1], tors[tor][1])):
                        matched.append(tor)
            activeCompatibles[tee] = matched
    return activeCompatibles


# Mentors Analysis by ID
mentors = groupAnalysis(mentors_active, mentors_file)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end = " ")
print("INFO: Finished analyzing mentor responses")

# Reponses Analysis by ID
mentees = groupAnalysis(mentees_active, mentees_file)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end = " ")
print("INFO: Finished analyzing mentee responses")

# match
activeMatched = matches(mentors, mentees)

print("\n")
print(activeMatched)

    


