
#####Task 4#####

def my_split(sentence, separator):
    sentenceList = []
    word = ""
    for i in sentence:
        if i == separator:
            sentenceList.append(word)
            word = ""
        else:
            word += i
    sentenceList.append(word)
    return sentenceList

def my_join(words, separator):

    result = ""
    for i in range(len(words)):
        result += words[i]
        if i < len(words) - 1:
            result += separator
    return result

sentence = input("Please enter sentence: ")

words = my_split(sentence, " ")

print(my_join(words, ","))

for word in words:
    print(word)

