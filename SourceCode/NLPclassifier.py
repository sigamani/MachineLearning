import numpy as np
import nltk
from nltk.corpus import state_union
import difflib
import re 
import sys


# Intermediate helper functions
def uppercase(matchobj):
    return matchobj.group(0).upper()

def capitalize(s):
    return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, s)

def cleanUpPunctuation(s):
    s = s.replace("( ", "(")


#Main method using pos-tagging
def Method(text):

    tagged_sent = nltk.pos_tag([word.lower() for word in nltk.word_tokenize(text)])
    normalized_sent = [w.capitalize() if t in ["NN","NNS"] else w for (w,t) in tagged_sent]    
    normalized_sent = re.sub(r" (?=[\.,!?:;])", "", ' '.join(normalized_sent))

    normalized_sent = normalized_sent.replace("( ", "(")
    normalized_sent = normalized_sent.replace(" )", ")")
    normalized_sent = normalized_sent.replace("[ ", "[")
    normalized_sent = normalized_sent.replace(" ]", "]")
    normalized_sent = normalized_sent.replace("{ ", "{")
    normalized_sent = normalized_sent.replace(" }", "}")
    normalized_sent = normalized_sent.replace("`` ", "\"")
    normalized_sent = normalized_sent.replace(r" ''", "\"")
    normalized_sent = normalized_sent.replace(r" '", "\'")

    # capitalise first word of each sentence after first pass 
    result = capitalize(normalized_sent)      
           
    return result



def diff_letters(a,b):

    return sum ( a[i] != b[i] for i in range(len(a)) )

def count_letters(word):
    BAD_LETTERS = " (?=[\.,'!?:;\n])"
    return len([letter for letter in word if letter not in BAD_LETTERS])


def Accuracy(a,b):

    value = -999.999
    print(len(a))
    print(len(b))

    if (len(a) != len(b)):
        print("Input/output text are the same size (cannot compute accuracy).")
    else:

        #get count of different characters
        diff_characters =  sum ( a[i] != b[i] for i in range(len(a)) )

        #get total count
        BAD_LETTERS = " (?=[\.,'!?:;\n])"
        total_letters = len([letter for letter in b if letter not in BAD_LETTERS])

        accuracy = 1 - np.float(diff_characters) / np.float(total_letters)

    return accuracy


def Scenario1():

    # Easy 2 sentences taken from BBC website
    Input = """MAYOR OF LONDON SADIQ KHAN WARNED THE RETIRED COURT OF APPEAL JUDGE MUST URGENTLY IMPROVE RELATIONS WITH LOCAL RESIDENTS. THE FIRE ON 14 JUNE IS THOUGHT TO HAVE KILLED AT LEAST 80 PEOPLE, ALTHOUGH POLICE SAY THE FINAL TOLL WILL NOT BE KNOWN UNTIL AT LEAST THE END OF THE YEAR."""
    expected = """Mayor of London Sadiq Khan warned the retired Court of Appeal judge must urgently improve relations with local residents. The fire on 14 June is thought to have killed at least 80 people, although police say the final toll will not be known until at least the end of the year."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def Scenario2():

    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND HAVE BEEN HIT BY IT FAILURE, CAUSED BY A LARGE-SCALE CYBER-ATTACK. LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE (IPL)."""
    expected = """NHS services across England and some in Scotland have been hit by IT failure, caused by a large-scale cyber-attack. Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. Police have arrested three people suspected of illegal betting in the Indian Premier League (IPL)."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def Scenario3():

    Input = state_union.raw("2006-GWBush.txt").upper()
    expected = state_union.raw("2006-GWBush.txt")

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def Scenario4(): # Double Quotations

    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND HAVE BEEN "HIT BY IT FAILURE" SAID TOM, CAUSED BY A LARGE-SCALE CYBER-ATTACK. LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE {IPL}."""
    expected = """NHS services across England and some in Scotland have been "hit by IT failure" said Tom, caused by a large-scale cyber-attack. Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. Police have arrested three people suspected of illegal betting in the Indian Premier League {IPL}."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)

def Scenario5(): # Single Quotations for plural

    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND HAVE BEEN HIT BY IT FAILURE, CAUSED BY A LARGE-SCALE CYBER-ATTACK. LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. THIS IS A RANDOM CHORUS OF HELLO'S. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE (IPL)."""
    expected = """NHS services across England and some in Scotland have been hit by IT failure, caused by a large-scale cyber-attack. Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. This is a random chorus of hello's. Police have arrested three people suspected of illegal betting in the Indian Premier League (IPL)."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def Scenario6(): # Question mark as punctuation 

    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND HAVE BEEN HIT BY IT FAILURE, CAUSED BY A LARGE-SCALE CYBER-ATTACK? LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. THIS IS A RANDOM CHORUS OF HELLO'S. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE (IPL)."""
    expected = """NHS services across England and some in Scotland have been hit by IT failure, caused by a large-scale cyber-attack? Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. This is a random chorus of hello's. Police have arrested three people suspected of illegal betting in the Indian Premier League (IPL)."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def Scenario7(): # Single quotations for quotation marks (wont work with current method)

    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND HAVE BEEN 'HIT BY IT FAILURE' SAID TOM, CAUSED BY A LARGE-SCALE CYBER-ATTACK. LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE (IPL)."""
    expected = """NHS services across England and some in Scotland have been 'hit by IT failure' said Tom, caused by a large-scale cyber-attack. Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. Police have arrested three people suspected of illegal betting in the Indian Premier League (IPL)."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)

def Scenario8(): # Begin sentence with quote

    # Longer phrase
    Input = """NHS SERVICES ACROSS ENGLAND AND SOME IN SCOTLAND. "WE HAVE BEEN HIT BY IT FAILURE" SAID TOM, CAUSED BY A LARGE-SCALE CYBER-ATTACK. LEWIS HAMILTON AND VALTTERI BOTTAS SECURED A MERCEDES ONE-TWO IN SECOND PRACTICE AT THE SPANISH GRAND PRIX, COMFORTABLY CLEAR OF THE FERRARIS. POLICE HAVE ARRESTED THREE PEOPLE SUSPECTED OF ILLEGAL BETTING IN THE INDIAN PREMIER LEAGUE (IPL)."""
    expected = """NHS services across England and some in Scotland. "We have been hit by IT failure" said Tom, caused by a large-scale cyber-attack. Lewis Hamilton and Valtteri Bottas secured a Mercedes one-two in second practice at the Spanish Grand Prix, comfortably clear of the Ferraris. Police have arrested three people suspected of illegal betting in the Indian Premier League (IPL)."""

    result = Method(Input)
    print(result)

    p = Accuracy(result,expected)
    print("Accuracy: %f" % p)


def main():
    
    with open(sys.argv[1], 'r') as f:
        contents = f.read()
    print(Method(contents))

   #Run test scenarios
  # Scenario1()
  # Scenario2()
  # Scenario4()
  # Scenario5()
  # Scenario6()
   #Scenario7()
  # Scenario8()

if __name__ == "__main__":

   main()


  


