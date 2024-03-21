# adapted from https://github.com/CeeZh/LLoVi
from string import Template
import re

def first_char_as_answer(res):
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    if res[0] in mapping:
        return mapping[res[0]]
    return -1

def identity(res):
    return res

def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            #pred_letter = res[anchor_index+len(anchor)]
            pred_letter = res[anchor_index+len(anchor):].strip()[0]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred
    return f

def string_after_anchor(anchor):
    def f(res):
        anchor_index = res.find(anchor)
        pred_string = ""  # if decoding failed, return ""
        if anchor_index >= 0:
            pred_string = res[anchor_index+len(anchor):]
        return pred_string
    return f

def get_intervals_as_list(text):
    text = text.split('.')[0]
    text = text.strip()
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index+1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        start, end = int(start_text), int(end_text)
        intervals.append([start, end])
    return intervals

def get_intervals_as_list_after_anchor(anchor):
    def f(text_in):
        anchor_index = text_in.find(anchor)
        text = ""  # if decoding failed, return ""
        if anchor_index >= 0:
            text = text_in[anchor_index+len(anchor):]

        pattern = r'(?<=\[).+?(?=\])' 
        intervals_in = re.findall(pattern, text)
        intervals_in = list(set(intervals_in))
        if len(intervals_in) == 0:
            intervals = [[0,0]]
        else:
            intervals = []
            for i in intervals_in:
                try:
                    st, end = i.split(', ')
                    intervals.append([int(st),int(end)])
                except:
                    continue
            if len(intervals) == 0: intervals = [[0,0]]
        return intervals
    return f


class PromptTemplate(object):
    def __init__(self, head, template, post_process_fn, max_new_tokens):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn
        self.max_new_tokens = max_new_tokens

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        # match variable names: duration, narration, question, optionA, optionB, optionC, optionD, optionE, num_words
        prompt_filled = []
        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled

    def fill_each(self, prompt_idx, **kwargs):
        return self.prompt_template[prompt_idx].substitute(kwargs)


class PromptFactory(object):
    def __init__(self):
        self.prompt_templates = self.build()
    
    def build(self):
        prompt_templates = {}

        ####################### GPT (from LLoVi) #######################

        ## 1. update input descriptions ##
        
        # egoschema LLoVi sum(q)
        prompt_templates['sum_q'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("You are given some language descriptions of a first person view video. The video is ${duration} seconds long. Each sentence "
                         "describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. "
                         "Here are the descriptions: ${narration}.\n Please give me a ${num_words} words summary. When doing summarization, remember that "
                         "your summary will be used to answer this multiple choice question: ${question}"),
            ],
            post_process_fn = identity,
            max_new_tokens = 500, #not used in gpt
        )

        ## 2. answer question (generative classifier) ##
        
        # egoschema QA (raw captions as input)
        prompt_templates['qa_standard'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                         "and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response "
                         "or explanation. You are given some language descriptions of a first person view video. The video is "
                         "${duration} seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential "
                         "and non-overlapping which cover the whole video exactly. Here are the descriptions: ${narration}.\n You are "
                         "going to answer a multiple choice question based on the descriptions, and your answer should be a single "
                         "letter chosen from the choices.\n Here is the question: ${question}.\n Here are the choices.\n "
                         "A: ${optionA}\n B: ${optionB}\n C: ${optionC}\n D: ${optionD}\n E: ${optionE}\n"),
            ],
            post_process_fn = first_char_as_answer,
            max_new_tokens = 20, #not used in gpt
        )
        # egoschema QA (summary as input)
        prompt_templates['qa_sum'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one "
                         "of the letters (A, B, C, D, or E). You must not provide any other response or explanation. You are given some language "
                         "descriptions of a first person view video. The video is ${duration} seconds long. Here are the descriptions: ${narration}.\n "
                         "You are going to answer a multiple choice question based on the descriptions, and your answer should be a single letter chosen "
                         "from the choices.\n Here is the question: ${question}.\n Here are the choices.\n "
                         "A: ${optionA}\n B: ${optionB}\n C: ${optionC}\n D: ${optionD}\n E: ${optionE}\n"),
            ],
            post_process_fn = first_char_as_answer,
            max_new_tokens = 20, #not used in gpt
        )
        # next-qa QA, intentQA QA
        prompt_templates['qa_next'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, and your answer must be one of the "
                         "letters (A, B, C, D, or E). You must not provide any other response or explanation. If you are not sure, answer with the most "
                         "likely answer. You are given some language descriptions of a first person view video. The video is 1 FPS and the descriptions are "
                         "the captions every 2 frames. Each caption starts with the frame number.\nHere are the descriptions:\n${narration}\n Here is the question: "
                         "${question}?\n Here are the choices:\n (A): ${optionA}\n (B): ${optionB}\n (C): ${optionC}\n (D): ${optionD}\n (E): ${optionE}\n"),
            ],
            post_process_fn = first_char_as_answer,
            max_new_tokens = 20, #not used in gpt
        )
        # next-gqa GQA
        prompt_templates['gqa'] = PromptTemplate(
            head = "You are a helpful expert in first person view video analysis.",
            template = [
                Template("I will provide video descriptions and one question about the video. The video is 1 FPS and the descriptions are the captions every "
                         "2 frames. Each caption starts with the frame number.\n To answer this question, what is the minimun frame interval to check?\n "
                         "Follow this format: [frame_start_index, frame_end_index]. Do not provide any explanation.\n Here are the descriptions:\n${narration}\n "
                         "Here is the question: ${question}?\n Please follow the output format as follows:\n #Example1: [5, 19]\n #Example2: [30, 60]\n "
                         "#Example3: [1, 10] and [50, 60]"),
            ],
            post_process_fn = get_intervals_as_list,
            max_new_tokens = 50, #not used in gpt
        )


        ####################### MISTRAL #######################
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        ## 1. update input descriptions ##

        # egoschema LLoVi sum(q) mistral
        anchor = E_INST
        prompt_templates['sum_q_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + "You are given some language descriptions of a first person view video. The video is ${duration} seconds long. Each sentence "
                         "describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions: ${narration}.\n Please give me a ${num_words} words summary. When doing summarization, remember that "
                         "your summary will be used to answer this multiple choice question: ${question}" + E_INST),
            ],
            post_process_fn = string_after_anchor(anchor),
            max_new_tokens = 500,
        )
        # egoschema LLoVi sum(q) w/ timestamp mistral
        anchor = E_INST
        prompt_templates['sum_q_tmstp_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + "You are given some language descriptions of a first person view video. The video is ${duration} seconds long. Each sentence "
                         "describes a ${clip_length}s clip. The descriptions are sequential and non-overlapping which cover the whole video exactly. Here are the descriptions with their timestamps: ${narration}.\n Please give me a ${num_words} words summary. When doing summarization, remember that "
                         "your summary will be used to answer this multiple choice question: ${question}" + E_INST),
            ],
            post_process_fn = string_after_anchor(anchor),
            max_new_tokens = 500,
        )
        # egoschema rephrase mistral
        anchor = " The rephrased list is as follows:\n"
        prompt_templates["rephrase_mistral"] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given a list of ${num_to_rephrase} language descriptions for a first person view video. Each sentence describes a ${clip_length}s clip. Here are the descriptions as a list:\n${memory}.\n"
                         "Please summarize and rephrase each item in the list as a single sentence of ${num_words_in_rephrase} words. Keep the same original subject (eg: #C, #O). Keep all information intact without leaving anything out. "
                         "Return only the rephrased list of ${num_to_rephrase} descriptions in the same order, without additional details. "
                         + E_INST + anchor),
            ],
            post_process_fn = string_after_anchor(anchor),
            max_new_tokens = 500,
        )
        # egoschema rephrase w/ timestamp mistral
        anchor = " The rephrased list is as follows:\n"
        prompt_templates["rephrase_tmstmp_mistral"] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given a list of ${num_to_rephrase} language descriptions for a video, together with timestamps:\n${memory}.\n"
                         "Please summarize and rephrase each entry in the list separately, using half the number of words. Keep the same original subject (eg: #C, #O). Mention timesteps or durations concisely when necessary. Keep all information intact without leaving anything out. Do not combine list entries. "
                         "Return only the rephrased list of exactly ${num_to_rephrase} entries in the same order, without additional details. "
                         + E_INST + anchor),
            ],
            post_process_fn = string_after_anchor(anchor),
            max_new_tokens = 1000,
        )
        # egoschema rephrase+sum(q) mistral
        anchor1 = " The rephrased list is as follows:\n"
        anchor2 = E_INST
        prompt_templates["rephrase_sum_mistral"] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given a list of ${num_to_rephrase} language descriptions for a first person view video. Each sentence describes a ${clip_length}s clip. Here are the descriptions as a list:\n${memory}.\n"
                         "Please summarize and rephrase each item in the list as a single sentence of ${num_words_in_rephrase} words. Keep the same original subject (eg: #C, #O). Keep all information intact without leaving anything out. "
                         "Return only the rephrased list of ${num_to_rephrase} descriptions in the same order, without additional details. "
                         + E_INST + anchor),
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given some language descriptions of a first person view video. The video is ${duration} seconds long. The descriptions are non-overlapping which cover the whole video exactly. Here are the descriptions: ${narration}.\n Please give me a ${num_words} words summary. When doing summarization, remember that "
                         "your summary will be used to answer this multiple choice question: ${question}" + E_INST),
            ],
            post_process_fn = [string_after_anchor(anchor1), string_after_anchor(anchor2)],
            max_new_tokens = 500,
        )
        # egoschema rephrase+sum(q) w/ timestamp mistral
        anchor1 = " The rephrased list is as follows:\n"
        anchor2 = E_INST
        prompt_templates["rephrase_tmstp_sum_mistral"] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given a list of ${num_to_rephrase} language descriptions for a video, together with timestamps:\n${memory}.\n"
                         "Please summarize and rephrase each entry in the list separately, using half the number of words. Keep the same original subject (eg: #C, #O). Mention timesteps or durations concisely when necessary. Keep all information intact without leaving anything out. Do not combine list entries. "
                         "Return only the rephrased list of exactly ${num_to_rephrase} entries in the same order, without additional details. "
                         + E_INST + anchor),
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS + 
                         "You are given some language descriptions of a first person view video. The video is ${duration} seconds long. The descriptions are non-overlapping which cover the whole video exactly. Here are the descriptions: ${narration}.\n Please give me a ${num_words} words summary. When doing summarization, remember that "
                         "your summary will be used to answer this multiple choice question: ${question}" + E_INST),
            ],
            post_process_fn = [string_after_anchor(anchor1), string_after_anchor(anchor2)],
            max_new_tokens = 1000,
        )

        ## 2. answer question (generative classifier) ##

        # egoschema QA (raw captions as input) mistral
        anchor = E_INST
        prompt_templates['qa_standard_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS +
                         "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                         "and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response "
                         "or explanation. You are given some language descriptions of a first person view video. The video is "
                         "${duration} seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential "
                         "and non-overlapping which cover the whole video exactly. Here are the descriptions: ${narration}.\n You are "
                         "going to answer a multiple choice question based on the descriptions, and your answer should be a single "
                         "letter chosen from the choices.\n Here is the question: ${question}.\n Here are the choices.\n "
                         "A: ${optionA}\n B: ${optionB}\n C: ${optionC}\n D: ${optionD}\n E: ${optionE}\n" + E_INST),
            ],
            post_process_fn = first_char_after_anchor(anchor),
            max_new_tokens = 20,
        )
        # egoschema QA (raw captions as input w/ timestamps) mistral
        prompt_templates['qa_tmstp_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS +
                         "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                         "and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response "
                         "or explanation. You are given some language descriptions of a first person view video. The video is "
                         "${duration} seconds long. Each sentence describes a ${clip_length}s clip. The descriptions are sequential "
                         "and non-overlapping which cover the whole video exactly. Here are the descriptions with their timestamps: ${narration}.\n You are "
                         "going to answer a multiple choice question based on the descriptions, and your answer should be a single "
                         "letter chosen from the choices.\n Here is the question: ${question}.\n Here are the choices.\n "
                         "A: ${optionA}\n B: ${optionB}\n C: ${optionC}\n D: ${optionD}\n E: ${optionE}\n" + E_INST),
            ],
            post_process_fn = first_char_after_anchor(anchor),
            max_new_tokens = 20,
        )
        # egoschema QA (summary as input) mistral
        anchor = E_INST
        prompt_templates['qa_sum_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS +
                         "Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question, "
                         "and your answer must be one of the letters (A, B, C, D, or E). You must not provide any other response "
                         "or explanation. You are given some language descriptions of a first person view video. The video is "
                         "${duration} seconds long. Here are the descriptions: ${narration}.\n You are "
                         "going to answer a multiple choice question based on the descriptions, and your answer should be a single "
                         "letter chosen from the choices.\n Here is the question: ${question}.\n Here are the choices.\n "
                         "A: ${optionA}\n B: ${optionB}\n C: ${optionC}\n D: ${optionD}\n E: ${optionE}\n" + E_INST),
            ],
            post_process_fn = first_char_after_anchor(anchor),
            max_new_tokens = 20,
        )
        # next-gqa GQA mistral
        anchor = E_INST
        prompt_templates['gqa_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template(B_INST + B_SYS + "You are a helpful expert in first person view video analysis. " + E_SYS +
                         "I will provide video descriptions and one question about the video. The video is 1 FPS and the descriptions are the captions every "
                         "2 frames. Each caption starts with the frame number.\n To answer this question, what is the minimun frame interval to check?\n "
                         "Follow this format: [frame_start_index, frame_end_index]. Always provide an interval. If not sure, give your best guess. Do not provide any explanation.\n Here are the descriptions:\n${narration}\n "
                         "Here is the question: ${question}?\n Please follow the output format as follows:\n #Example1: [5, 19]\n #Example2: [30, 60]\n "
                         "#Example3: [1, 10] and [50, 60]" + E_INST),
            ],
            #post_process_fn = get_intervals_as_list,
            post_process_fn = get_intervals_as_list_after_anchor(anchor),
            max_new_tokens = 100,
        )

        ## 2. answer question (log-likelihood classifier) ##

        # egoschema QA mistral (log-likelihood eval)
        prompt_templates['qa_ll_mistral'] = PromptTemplate(
            head = "",
            template = [
                Template("${narration} ${question}"),
                Template(" ${answer}"),
            ],
            post_process_fn = identity,
            max_new_tokens = 20,
        )
        # next-qa QA mistral (log-likelihood eval)
        prompt_templates['qa_ll_mistral_nextqa'] = PromptTemplate(
            head = "",
            template = [
                Template("${narration} Based on the description above, answer the following question: ${question}? Select one of these choices as the answer:\nA: ${optionA}\nB: ${optionB}\nC: ${optionC}\nD: ${optionD}\nE: ${optionE}\nThe correct answer is, "),
                Template("${answer_id}: ${answer}"),
            ],
            post_process_fn = identity,
            max_new_tokens = 20,
        )
        
        return prompt_templates

    def get(self, prompt_type):
        return self.prompt_templates[prompt_type]
