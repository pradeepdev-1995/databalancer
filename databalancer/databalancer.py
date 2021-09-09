import                                              pandas as pd
from databalancer.paraphraseGeneratorClient import               paraPharaseGenerator
from databalancer.paraphraseGeneratorClient import               modelAndTokenizerInitializer
from databalancer.paraphraseInputGeneratorClient import          paraphraseInputSentenceGenerator


import matplotlib.pyplot                            as plt

def balanceDataset(dataset_name,saveAsCsv=True,pretrained_model="ramsrigouthamg/t5_paraphraser",pretrained_tokenizer="t5-base",seed=42):
    data                                            = pd.read_csv(dataset_name)
    model,tokenizer,device                          = modelAndTokenizerInitializer(pretrained_model,pretrained_tokenizer,seed)
    columnList                                      = list()
    for col in data.columns:
        columnList.append(col)

    text_column                                     = columnList[0]
    class_column                                    = columnList[1]

    dataOriginal                                    = data

    value_dict                                      = data[class_column].value_counts().to_dict()

    balanced_flag                                   = len(list(set(list(value_dict.values())))) == 1

    while not (balanced_flag):
        balanceCountDict                            = dict()
        max_key                                     = max(value_dict, key=value_dict.get)
        max_count                                   = value_dict[max_key]
        value_dict.pop(max_key)

        for key, value in value_dict.items():
            balanceCountDict[key]                   = max_count - value

        for key, value in balanceCountDict.items():
            if (value != 0):
                inputSentenceList                   = paraphraseInputSentenceGenerator(data,class_column,text_column,key)

                if (value < 5):
                    each_para_count                 = 1
                    inputSentenceList               = inputSentenceList[:value]
                else:
                    each_para_count                 = int(value / 5)

                paraQuestionlist                    = []

                for sentence in inputSentenceList:
                    paraQuestionlist                = paraPharaseGenerator(sentence,each_para_count,model,tokenizer,device)

                paraFrame                           =   {
                                                        text_column: paraQuestionlist,
                                                        class_column: key
                                                        }

                each_df                             = pd.DataFrame(paraFrame, columns=[text_column, class_column])
                dataOriginal                        = pd.concat([dataOriginal, each_df], ignore_index=True)
            else:
                pass

        value_dict                                  = dataOriginal[class_column].value_counts().to_dict()

        balanced_flag                               = len(list(set(list(value_dict.values())))) == 1


    if(saveAsCsv):
        outfile                                     = "balanced_data.csv"
        dataOriginal.to_csv(outfile, index=False)
        return                                          True
    else:
        return                                          dataOriginal


def classCountVisualization(dataset_name):
    data                                            = pd.read_csv(dataset_name)

    columnList                                      = list()
    for col in data.columns:
        columnList.append(col)

    class_column                                    = columnList[1]

    pie_plot                                        = data[class_column].value_counts()
    pie_plot.plot(kind='pie')

    plt.show(block=True)
    plt.interactive(False)


