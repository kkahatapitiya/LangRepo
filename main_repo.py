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
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, CLIPTextModel
from sentence_transformers import SentenceTransformer

from merge import find_similarities, group_captions, init_generator
from util import clean_text


torch.manual_seed(42)
CLIP_VERSION = 'openai/clip-vit-large-patch14'
SENTENCE_T5_VERSION = 'sentence-transformers/sentence-t5-xl'

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = init_generator(device)
    # get text-encoder
    if args.text_encoder == 'clip':
        text_tokenizer = AutoTokenizer.from_pretrained(CLIP_VERSION)
        text_encoder = CLIPTextModel.from_pretrained(CLIP_VERSION)
    else:
        text_encoder = SentenceTransformer(SENTENCE_T5_VERSION).to(device)

    # answer
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        clip_length = int(1/args.fps) if args.fps < 1 else 1/args.fps

        all_narations = item['raw_naration']
        len_full = len(all_narations)
        del item['raw_naration']

        # for debugging
        if i==0:
            if 'quid' in item: 
                print(item['quid'])
            else: 
                print(item['uid'])

        ukey_name = 'quid' if 'quid' in item else 'uid'
        ukey = item[ukey_name]
        processed[ukey] = item
        processed[ukey]['prompt_template'] = []
        processed[ukey]['response'] = []
        processed[ukey]['pred_ij_rep'] = []
        processed[ukey]['pred_ij'] = []
        processed[ukey]['pred_i'] = []
        processed[ukey]['pred'] = []

        
        all_narations_i = all_narations
        prev_repo_keys_i = None

        # write iteration (indexed by i)
        for iter_i in range(args.num_iterations):

            repo_i = {}
            num_ch_i = args.num_chunks[iter_i] # num chunks in current iteration
            len_ch_i = len(all_narations_i)//num_ch_i # chunk size
            
            # if timestamp to be used, format so in first iteration only
            is_first_iter = True if iter_i==0 else False
            
            # chunks in each write iteration (indexed by j)
            for ch_j in range(num_ch_i):
                all_narations_ij = all_narations_i[len_ch_i*ch_j : len_ch_i*(ch_j+1)]
                item_ij = item.copy()
                del item_ij['narration']

                # get text embeddings
                if args.text_encoder == 'clip':
                    caption_tokens = text_tokenizer(all_narations_ij, padding=True, truncation=True, return_tensors="pt")
                    caption_emb = text_encoder(**caption_tokens).pooler_output.unsqueeze(0)
                else:
                    caption_emb = text_encoder.encode(all_narations_ij)
                    caption_emb = torch.from_numpy(caption_emb).unsqueeze(0)

                # find similar caption indices for grouping
                unm_idx, src_idx, srcdst_idx, dst_idx = find_similarities(metric=caption_emb, w=len_ch_i, sx=args.dst_stride,
                                                                          r=int(len_ch_i * args.merge_ratio), generator=generator)
                # group similar captions
                grouped_cap = group_captions(all_nar=all_narations_ij, offset=len_ch_i*ch_j, key_map=prev_repo_keys_i,
                                            unm_idx=unm_idx, src_idx=src_idx, srcdst_idx=srcdst_idx, dst_idx=dst_idx, debug=args.debug)
                
                repo_ij = {}
                repo_idx = []
                repo_ij_str = ''
                num_to_rephrase = 0 # counter to track num of groups to rephrase
                for mk, mv in grouped_cap.items():
                    if len(mv)>1: # multiple captions in a group
                        num_to_rephrase += 1
                        mv = sorted(mv, key=lambda k: int(str(k[1]).split(',')[0]))
                        key_k = ",".join([str(k[1]) for k in mv])
                        repo_idx.append(key_k)

                        # create list to rephrase
                        if args.use_tmstp and is_first_iter: # format with timestamps
                            repo_ij_str += f'{num_to_rephrase}. {". ".join(f"[t={k[1]}]: {k[0]}" for k in mv)}.\n'
                        else:
                            repo_ij_str += f'{num_to_rephrase}. {". ".join(f"{k[0]}" for k in mv)}.\n'
                        repo_ij[key_k] = mv

                    else: # single caption in a group
                        if args.use_tmstp and is_first_iter: # format with timestamps
                            if args.use_occ: # format with occurances
                                repo_ij[str(mk)] = f"At t={mv[0][1]}, {mv[0][0]} (x 1)."
                            else:
                                repo_ij[str(mk)] = f"At t={mv[0][1]}, {mv[0][0]}."
                        else:
                            if args.use_occ: # format with occurances
                                repo_ij[str(mk)] = f"{mv[0][0]} (x 1)."
                            else:
                                repo_ij[str(mk)] = f"{mv[0][0]}."
                        
                
                repo_ij_str = clean_text(repo_ij_str)
                item_ij['memory'] = repo_ij_str
                item_ij['num_to_rephrase'] = num_to_rephrase

                # fill rephrasing template
                prompt_ij = [prompter.fill_each(**item_ij, prompt_idx=0, fps=args.fps, clip_length=clip_length, 
                                                num_words=args.num_words_in_sum, num_words_in_rephrase=args.num_words_in_rephrase)]
                model.set_post_process_fn(prompter.post_process_fn[0])

                # call LLM for rephrasing
                pred_ij, info_ij = model.forward(prompter.head, prompt_ij, max_new_tokens=prompter.max_new_tokens)
                pred_ij = pred_ij.replace('\n\n', '\n').strip('\n').split('\n')

                for mk, mv in zip(repo_idx, pred_ij[:len(repo_idx)]):
                    # remove list numbers in rephrased output
                    if args.use_occ: # format with occurances
                        repo_ij[mk] = f'{mv[2:]} (x {len(mk.split(","))})'
                    else:
                        repo_ij[mk] = mv[2:]
                # mismatch in input-output groups: remove groups not in the output
                if len(repo_idx)>len(pred_ij):
                    print(f'WARNING!!... ({len(pred_ij)}/{len(repo_idx)}) - {ukey}-iter({iter_i})-ch({ch_j}):\n{pred_ij}')
                    for mk in repo_idx[-(len(repo_idx)-len(pred_ij)):]: del repo_ij[mk]
                
                repo_ij = OrderedDict(sorted(repo_ij.items(), key=lambda k: int(k[0].split(',')[0])))
                rephrased_repo_ij = (". ".join(repo_ij.values())+'.')
                rephrased_repo_ij = clean_text(rephrased_repo_ij)
                repo_i.update(repo_ij)

                processed[ukey]['prompt_template'].append(prompter.get_template_str()[0])
                processed[ukey]['response'].append(info_ij['response'])
                processed[ukey]['pred_ij_rep'].append(pred_ij)
                processed[ukey]['pred_ij'].append(rephrased_repo_ij)
            
            repo_i = OrderedDict(sorted(repo_i.items(), key=lambda k: int(k[0].split(',')[0])))
            rephrased_repo_i = (". ".join(repo_i.values())+'.')
            rephrased_repo_i = clean_text(rephrased_repo_i)
            processed[ukey]['pred_i'].append(rephrased_repo_i)

            all_narations_i = list(repo_i.values())
            prev_repo_keys_i = list(repo_i.keys())

        # multi-scale summaries
            
        # get duration of each scale
        ch_lens = []
        for rd_ch_j in args.num_chunks:
            ch_lens += [len_full//rd_ch_j]*rd_ch_j

        for rd_scl_idx, rd_scl in enumerate(args.read_scales): 
            item['narration'] = processed[ukey]['pred_ij'][rd_scl]
            item['duration'] = ch_lens[rd_scl_idx]

            # fill summarizing template
            prompt = [prompter.fill_each(**item, prompt_idx=1, fps=args.fps, num_words=args.num_words_in_sum)]
            model.set_post_process_fn(prompter.post_process_fn[1])
            # call LLM for summarizing
            pred, _ = model.forward(prompter.head, prompt, max_new_tokens=prompter.max_new_tokens)
            pred = clean_text(pred.replace('\n\n', '\n').strip('\n'))
            processed[ukey]['pred'].append(pred)
                
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

