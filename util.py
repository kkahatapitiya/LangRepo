# adapted from https://github.com/CeeZh/LLoVi
import pickle
import json
from pathlib import Path
import argparse
import torch


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_text(txt):
    return txt.replace(' .', '.').replace('...', '.').replace('..', '.').replace('  ', ' ')

def loglikelihood_classifier(logits, labels):
    """Calculate the loglikelihood of the model predictions given the labels"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    loglikelihood = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loglikelihood = loglikelihood.view(shift_logits.size(0), -1).sum(-1) / (shift_labels != -100).sum(-1)
    return loglikelihood

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='egoschema', type=str)         # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'
    parser.add_argument("--data_path", default='data/egoschema/lavila_subset.json', type=str) 
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)  
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 
    parser.add_argument("--fps", default=1.0, type=float) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    ## backup pred
    parser.add_argument("--backup_pred_path", default="", type=str)
    ## fewshot
    parser.add_argument("--fewshot_example_path", default="", type=str) 
    ## nextgqa
    parser.add_argument("--nextgqa_gt_ground_path", default="", type=str)
    parser.add_argument("--nextgqa_pred_qa_path", default="", type=str)

    # output
    parser.add_argument("--output_base_path", required=True, type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # prompting
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)       # Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="qa_standard", type=str)
    parser.add_argument("--task", default="qa", type=str)                   # sum, qa, gqa
    ## sum
    parser.add_argument("--num_words_in_sum", default=500, type=int)  

    # other
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=5, type=int)
    
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--text_encoder", default="clip", type=str)         # clip, sentence-t5

    parser.add_argument("--num_iterations", default=1, type=int)            # 1, 2
    parser.add_argument('--num_chunks',  default="[4]", type=json.loads)    # [4], [2]
    parser.add_argument("--merge_ratio", default=0.25, type=float)          # 0.25, 0.5
    parser.add_argument("--dst_stride", default=4, type=int)                # 4, 2
    parser.add_argument("--num_words_in_rephrase", default=20, type=int)
    parser.add_argument('--read_scales',  default="[-1]", type=json.loads)  # [-1], [-3,-2,-1]
    parser.add_argument("--use_tmstp", action='store_true')
    parser.add_argument("--use_occ", action='store_true')


    return parser.parse_args()


'''def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars'''
    
    
    