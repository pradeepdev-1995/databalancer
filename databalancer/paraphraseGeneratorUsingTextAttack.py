import textattack
from textattack.augmentation import EasyDataAugmenter
easy_aug = EasyDataAugmenter()


def paraPharaseGeneratorTextAttack(sentence):
    paraQuestionlist                = []
    paraQuestionlist                = easy_aug.augment(sentence)
    print("paraQuestionlist")
    print(paraQuestionlist)
    print("type")
    print(type(paraQuestionlist))
    print("length")
    print(len(paraQuestionlist))
    return paraQuestionlist