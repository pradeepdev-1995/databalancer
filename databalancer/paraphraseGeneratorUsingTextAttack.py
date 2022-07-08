import textattack
from textattack.augmentation import EasyDataAugmenter



def paraPharaseGeneratorTextAttack(sentence,each_para_count):
    easy_aug                        = EasyDataAugmenter(transformations_per_example=each_para_count)
    paraQuestionlistActual          = []
    paraQuestionlist                = easy_aug.augment(sentence)
    res                             = isinstance(paraQuestionlist, str)
    if (res):
        paraQuestionlistActual.append(paraQuestionlist)
    else:
        paraQuestionlistActual      = paraQuestionlist
    return paraQuestionlistActual