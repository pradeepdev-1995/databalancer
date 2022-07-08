import                                                 pandas as pd
from databalancer.paraphraseGeneratorClient import                  paraPharaseGeneratorT5
from databalancer.paraphraseGeneratorClient import                  modelAndTokenizerInitializer
from databalancer.paraphraseInputGeneratorClient import             paraphraseInputSentenceGenerator
from databalancer.paraphraseGeneratorUsingNlpAug import             paraPharaseGeneratorNlpAug
from databalancer.paraphraseGeneratorUsingTextAttack import         paraPharaseGeneratorTextAttack
import matplotlib.pyplot                               as plt

'''
Datset balancer function
1 - identify the column names from an input dataset
2 - Find the class with maximum text count
3 - Identify the number of texts required for each class to meet the maximum value
4 - Using t5_paraphraser generate as many as texts for each class to meet the maximum value
5 - Depends on the saveAsCsv value,store the balanced dataset as balanced_data.csv to local machine or return the balanced pandas
    dataframe to user
'''
def balanceDataset(dataset_name,saveAsCsv=True,balance_method=1,quantize=False,seed=42,model="bert-base-uncased"):
    data                                            = pd.read_csv(dataset_name)
    sentence                                        = ""
    columnList                                      = list()
    for col in data.columns:
        columnList.append(col)

    text_column                                     = columnList[0]
    class_column                                    = columnList[1]

    dataOriginal                                    = data

    value_dict                                      = data[class_column].value_counts().to_dict()

    balanced_flag                                   = len(list(set(list(value_dict.values())))) == 1
    print("Balancing started ..")
    iteration_count                                 = 0
    if (balance_method == 2):
        pretrained_model                            = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
        print("Balancing using \"ramsrigouthamg/t5-large-paraphraser-diverse-high-quality\" T5 model ")
        model, tokenizer, device                    = modelAndTokenizerInitializer(pretrained_model, quantize, seed)
    elif(balance_method == 3):
        print("Balancing using NLPAUG")
    elif(balance_method == 4):
        print("Balancing using TextAttack")
    else:
        pretrained_model = "ramsrigouthamg/t5_paraphraser"
        print("Balancing using \"ramsrigouthamg/t5_paraphraser\" T5 model ")
        model, tokenizer, device                    = modelAndTokenizerInitializer(pretrained_model, quantize, seed)


    while not (balanced_flag):
        iteration_count += 1
        if(iteration_count%10==0):
            print("Balancing iteration " + str(iteration_count) + "...")
        else:
            pass
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

                if(balance_method==3):
                    for sentence in inputSentenceList:
                        eachparaQuestionlist        = paraPharaseGeneratorNlpAug(sentence, each_para_count, model)
                        paraQuestionlist.extend(eachparaQuestionlist)
                elif(balance_method==4):
                    for sentence in inputSentenceList:
                        eachparaQuestionlist        = paraPharaseGeneratorTextAttack(sentence,each_para_count)
                        paraQuestionlist.extend(eachparaQuestionlist)
                else:
                    for sentence in inputSentenceList:
                        eachparaQuestionlist        = paraPharaseGeneratorT5(sentence,each_para_count,model,tokenizer,device)
                        paraQuestionlist.extend(eachparaQuestionlist)

                if (len(paraQuestionlist)==0):
                    paraQuestionlist.append(sentence)
                else:
                    pass
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

    print("Dataset balancing completed")

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


