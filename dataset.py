# adapted from https://github.com/CeeZh/LLoVi
## TODO(@kumarak): correct padding, min captions for rephrase
from torch.utils.data import Dataset
import pandas as pd
from util import load_json, parse_args, clean_text


class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        '''num_examples_to_run < 0: run all'''
        self.args = args
        self.narrations = self.get_descriptions()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        self.durations = load_json(args.duration_path)  # uid --> float
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run):
        if quids_to_exclude is not None:
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('uid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = '. '.join(narr)
        narr = clean_text(narr)
        return narr

    def get_anno(self):
        anno = load_json(self.args.anno_path)  # uid --> {question, option 0, option 1, option 2, option 3, option 4, truth (optional)}
        return anno

    def build(self):
        data = []
        for uid, item in self.anno.items():
            if uid not in self.narrations:
                continue
            narration = self.format_narration(self.narrations[uid])
            question = item['question']
            choices = [item['option 0'], item['option 1'], item['option 2'], item['option 3'], item['option 4']] 
            truth = item['truth'] if 'truth' in item else -1
            duration = int(self.durations[uid])
            data.append({
                'uid': uid,
                'narration': narration,
                'raw_naration': self.narrations[uid],
                'question': question,
                'optionA': choices[0],
                'optionB': choices[1],
                'optionC': choices[2],
                'optionD': choices[3],
                'optionE': choices[4],
                'truth': truth,
                'duration': duration,
            })
        return data


class NextDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('quid')
        self.min_captions = 16
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            #caption_every = int(1/self.args.fps)
            #narr = '.\n'.join([f'{int(i*caption_every)}: {cap}' for i, cap in enumerate(narr[::caption_every])])
            narr = '. '.join(narr)
        narr = clean_text(narr)
        return narr

    def get_anno(self):
        return pd.read_csv(self.args.anno_path)  # video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
         
    def build(self):
        data = []
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            uid = str(row['video'])
            quid = f"{row['video']}_{row['qid']}"
            if uid in self.narrations:
                id_to_use = uid
                
                narr = self.narrations[id_to_use]
                if len(narr) < self.min_captions: #if num_captions is too small for main_rephrase.py
                    narr_pad = narr + narr[-(self.min_captions - len(narr)): ] #padding
                    self.narrations[id_to_use] = narr_pad
                
            elif quid in self.narrations:
                id_to_use = quid
            else:
                continue

            question, truth = row['question'], row['answer']
            qid, q_type = row['qid'], row['type']
            choices = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            narration = self.format_narration(self.narrations[id_to_use])
            duration = int(self.durations[uid])
            data.append({
                'quid': quid,
                'uid': uid,
                'qid': qid,
                'q_type': q_type,
                'narration': narration,
                'raw_naration': self.narrations[id_to_use],
                'question': question,
                'optionA': choices[0],
                'optionB': choices[1],
                'optionC': choices[2],
                'optionD': choices[3],
                'optionE': choices[4],
                'truth': truth,
                'duration': duration,
            })
        return data


def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1):
    if args.dataset == 'egoschema':
        return EgoSchemaDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    else:
        return NextDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    print(len(dataset))
