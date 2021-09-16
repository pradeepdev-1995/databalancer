
'''
Module for select the sentences for generating paraphrases
'''

def paraphraseInputSentenceGenerator(data,class_column,text_column,key):
    inputSentenceList                       = []

    currentClassFrame                       = data[data[class_column] == key]
    questionList                            = currentClassFrame[text_column].to_list()

    inputSentenceList.append(questionList[0])
    inputSentenceList.append(questionList[-1])
    inputSentenceList.append(questionList[int(len(questionList) / 2)])
    inputSentenceList.append(questionList[int(int(len(questionList) / 2) / 2)])
    inputSentenceList.append(questionList[int((int(len(questionList) / 2) + len(questionList)) / 2)])

    return inputSentenceList
