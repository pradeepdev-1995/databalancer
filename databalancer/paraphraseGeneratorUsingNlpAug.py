import nlpaug.augmenter.word as naw

def paraPharaseGeneratorNlpAug(sentence, each_para_count, model):
    paraQuestionlistActual      = []
    aug                         = naw.ContextualWordEmbsAug(model_path=model, action="insert")
    paraQuestionlist            = aug.augment(sentence, n=each_para_count)
    res                         = isinstance(paraQuestionlist, str)
    if(res):
        paraQuestionlistActual.append(paraQuestionlist)
    else:
        paraQuestionlistActual = paraQuestionlist
    return paraQuestionlistActual