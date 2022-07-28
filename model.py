#!pip install ohmeow-blurr -q
#!pip install bert-score -q
#!pip install sacremoses

import pandas as pd
from fastai.text.all import *
from transformers import *
#from blurr.data.all import *
#from blurr.modeling.all import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
#Get data
#Input the path to data
df = pd.read_csv('data.csv', error_bad_lines=False)
df = df.dropna().reset_index()

#Select part of data we want to keep
df = df[['summary','reviewText']]


#Select only part of it (makes testing faster)
articles = df.head(100)

#Import the pretrained model

pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model_name, 
                                                                  model_cls=BartForConditionalGeneration)

#Create mini-batch and define parameters
hf_batch_tfm = Seq2SeqBatchTokenizeTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
    task='summarization',
    text_gen_kwargs=
 {'max_length': 248,'min_length': 56,'do_sample': False, 'early_stopping': True, 'num_beams': 4, 'temperature': 1.0, 
  'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
 'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
 'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
 'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False})


#Prepare data for training
blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=hf_batch_tfm), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader('reviewText'), get_y=ColReader('summary'), splitter=RandomSplitter())
dls = dblock.dataloaders(articles, batch_size = 2)

seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'en' },
            'returns': ["precision", "recall", "f1"]
        }
    }

model = BaseModelWrapper(hf_model)
learn_cbs = [BaseModelCallback]
fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,
                splitter=partial(blurr_seq2seq_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()

learn.fit_one_cycle(1, lr_max=3e-5, cbs=fit_cbs)

#outputs = learn.blurr_generate(text, early_stopping=False, num_return_sequences=1)

#for idx, o in enumerate(outputs):
#    print(f'=== Prediction {idx+1} ===\n{o}\n')