import nlpaug.augmenter.word as naw

def paraPharaseGeneratorNlpAug(sentence, each_para_count, model):
    paraQuestionlist    = []
    aug                 = naw.ContextualWordEmbsAug(model_path=model, action="insert")
    paraQuestionlist    = aug.augment(sentence, n=each_para_count)
    print("paraQuestionlist")
    print(paraQuestionlist)
    return paraQuestionlist