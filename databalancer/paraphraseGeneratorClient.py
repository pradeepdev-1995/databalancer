import                                          torch
from transformers import                        T5ForConditionalGeneration,T5Tokenizer
from modelQuantization import                   quantizeModel

pretrained_model                                = "ramsrigouthamg/t5_paraphraser"
t5_paraphraser_model                            = quantizeModel(pretrained_model)
tokenizer_t5_paraphraser                        = T5Tokenizer.from_pretrained(pretrained_model)

pretrained_model                                = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
t5_paraphraser_model_high_quality               = quantizeModel(pretrained_model)
tokenizer_model_high_quality                    = T5Tokenizer.from_pretrained(pretrained_model)

device                                          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Generate paraphrase of a sentence using T5 paraphrase model.
Default parameters provided for the T5ForConditionalGeneration class are as below 
pad_to_max_length=True,
return_tensors="pt",
do_sample=True,
max_length=256,
top_k=120,
top_p=0.98,
early_stopping=True
'''

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def modelAndTokenizerInitializer(pretrained_model,seed):
    if(pretrained_model=="ramsrigouthamg/t5_paraphraser"):
        model                                   = t5_paraphraser_model
        tokenizer                               = tokenizer_t5_paraphraser
    elif(pretrained_model=="ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"):
        model                                   = t5_paraphraser_model_high_quality
        tokenizer                               = tokenizer_model_high_quality
    else:
        model                                   = T5ForConditionalGeneration.from_pretrained(pretrained_model)

    model                                       = model.to(device)
    set_seed(seed)
    return                                      model,tokenizer,device


def paraPharaseGeneratorT5(sentence,each_para_count,model,tokenizer,device,return_tensors="pt",do_sample=True,max_length=256,top_k=120,top_p=0.98,early_stopping=True):
    paraQuestionlist                            = []
    text                                        = "paraphrase: " + sentence

    encoding                                    = tokenizer.encode_plus(text, padding='longest', return_tensors=return_tensors)
    input_ids, attention_masks                  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    beam_outputs                                = model.generate(
                                                    input_ids               =   input_ids,
                                                    attention_mask          =   attention_masks,
                                                    do_sample               =   do_sample,
                                                    max_length              =   max_length,
                                                    top_k                   =   top_k,
                                                    top_p                   =   top_p,
                                                    early_stopping          =   early_stopping,
                                                    num_return_sequences    =   each_para_count
    )

    for beam_output in beam_outputs:
        sent                                    = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in paraQuestionlist:
            paraQuestionlist.append(sent)

    return paraQuestionlist