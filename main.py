# adapted from https://github.com/CeeZh/LLoVi
import os
from pathlib import Path
from util import *
from eval import *
from dataset import get_dataset
from prompts import PromptFactory
from model import get_model
from tqdm import tqdm
from pprint import pprint


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

    # get LLM
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)

    # answer
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        clip_length = int(1/args.fps) if args.fps < 1 else 1/args.fps

        # for visually-grounded vqa, format each caption with frame number
        if args.task == 'gqa':
            all_narations = item['raw_naration']
            all_narations = [f'{nii_idx+1}: {nii}' for nii_idx, nii in enumerate(all_narations)]
            if i==0: print(len(all_narations))
            item['narration'] = '. '.join(all_narations).replace('..', '.')
        del item['raw_naration']
        
        # fill template
        prompt = prompter.fill(**item, fps=args.fps, clip_length=clip_length, num_words=args.num_words_in_sum)
        # call LLM
        pred, info = model.forward(prompter.head, prompt, max_new_tokens=prompter.max_new_tokens)
        
        ukey_name = 'quid' if 'quid' in item else 'uid'
        ukey = item[ukey_name]
        processed[ukey] = item
        processed[ukey]['prompt_template'] = prompter.get_template_str()
        processed[ukey]['response'] = info['response']
        processed[ukey]['pred'] = pred
        if args.save_info:
            processed[ukey]['info'] = {k: v for k, v in info.items() if k != 'response'}
        del processed[ukey]['narration'] 

        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)

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
