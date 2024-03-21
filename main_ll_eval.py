# adapted from https://github.com/CeeZh/LLoVi
import os
from pathlib import Path
from util import *
from eval import *
from dataset import get_dataset
from prompts import PromptFactory
from tqdm import tqdm
from pprint import pprint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from util import loglikelihood_classifier


def launch():
    args = parse_args()
    pprint(args)

    # output
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)

    # resume
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if 'data' in processed:
            processed = processed['data']

    # get input
    quids_to_exclude = set(list(processed.keys()))
    dataset = get_dataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run)

    # configure prompt
    prompter = PromptFactory().get(args.prompt_type)

    # 4-bit quantization mixtral to manage gpu memory
    if 'mixtral' in args.model.lower():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get LLM
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 device_map="auto",
                                                 quantization_config=quantization_config,
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.padding_side = "left"

    # answer
    pbar = tqdm(total=len(dataset))
    correct, total = 0, 0

    for i, item in enumerate(dataset):
        item['narration'] = item['narration'].replace('\n\n', ' ').replace('\n', ' ')
        del item['raw_naration']
        
        # batch {question: answer_option} for each answer
        prompt_head = prompter.fill_each(**item, prompt_idx=0)
        prompt_tails = [prompter.fill_each(**item, prompt_idx=1, answer_id=op_i,
                                           answer=item[f'option{op_i}']) for op_i in ['A','B','C','D','E']]
        prompts = [prompt_head + pt_i for pt_i in prompt_tails]

        a_prompt_tokens = tokenizer(prompt_tails)
        a_prompt_lengths = [len(x) - 1 for x in a_prompt_tokens['input_ids']] # to filter only answer tokens
        qa_prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)

        # for debugging
        if i==0:
            if 'quid' in item:
                print(item['quid'])
            else: 
                print(item['uid'])
            print(prompt_head, prompt_tails)
        
        #feeding all options as a batch
        labels = qa_prompt_tokens['input_ids'].clone()
        for idx, length in enumerate(a_prompt_lengths):
            labels[idx, :-length] = -100
        model_inputs = {x: y.to(device) for x,y in qa_prompt_tokens.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs, labels=labels).logits.detach()
        
        # manage gpu memory by using bs=1 (feeding each option separately)
        '''max_tk = 4096
        labels = qa_prompt_tokens['input_ids'].clone()[:, -max_tk:]
        #print(qa_prompt_tokens['input_ids'].shape, labels.shape)
        for idx, length in enumerate(a_prompt_lengths):
            labels[idx, :-length] = -100
        model_outputs = []
        for ch_i in range(len(prompts)):
            model_inputs = {x: y[ch_i:ch_i+1, -max_tk:].to(device) for x,y in qa_prompt_tokens.items()}
            with torch.no_grad():
                model_outputs.append(model(**model_inputs, labels=labels[ch_i:ch_i+1]).logits.detach())
        model_outputs = torch.cat(model_outputs, dim=0)
        del qa_prompt_tokens, model_inputs'''
        
        # log-likelihood classifier
        loss = loglikelihood_classifier(model_outputs, labels)
        # select highest-probable option (argmin of CE loss)
        pred = loss.argmin().item()
        correct += (item['truth'] == pred)
        total += 1

        ukey_name = 'quid' if 'quid' in item else 'uid'
        ukey = item[ukey_name]
        processed[ukey] = item
        processed[ukey]['prompt_template'] = prompter.get_template_str()
        processed[ukey]['pred'] = pred
        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)
        torch.cuda.empty_cache()
    
    if total != 0:
        print(f"'acc': {correct/total}, 'num_corrects': {correct}, 'num_total': {total}, 'num_valids': {total}")

    # incorporate with backup prediction
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        if 'data' in backup:
            backup = backup['data']
        for uid in processed:
            if processed[uid]['pred'] == -1:
                processed[uid]['pred'] = backup[uid]['pred']

    # if eval
    if not args.disable_eval:
        if args.task == 'qa':
            if args.dataset == 'egoschema':
                processed = eval_qa_egoschema(processed)
            elif args.dataset in ['nextqa', 'intentqa', 'nextgqa']:
                processed = eval_qa_nextqa(args.anno_path, processed)
        elif args.task == 'gqa':
            if args.dataset == 'nextgqa':
                pred_qa_path = args.nextgqa_pred_qa_path if len(args.nextgqa_pred_qa_path) > 0 else None
                processed = eval_gqa(args.nextgqa_gt_ground_path, processed, pred_qa_path=pred_qa_path)
        elif args.task == 'sum':
            processed, sum_data = eval_sum(processed)
            save_json(sum_data, f'{Path(output_path).parent / Path(output_path).stem}_data.json')

    save_json(processed, output_path)
    


if __name__ == '__main__':
    launch()