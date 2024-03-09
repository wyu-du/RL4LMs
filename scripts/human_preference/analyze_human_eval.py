import json
import numpy as np

with open('human_eval_wanyu/human_eval_fd_model1.json') as f:
    lines = json.load(f)

coop_list, abs_list, engage_list, hallucinate_list, generic_list = [], [], [], [], []
for line in lines:
    coop_list.append(line['Q1']['Cooperativeness']['Score'])
    engage_list.append(line['Q1']['Engagingness']['Score'])
    abs_list.append(line['Q1']['Abstractiveness']['Score'])
    hallucinate_list.append(line['Q2']['Hallucination']['Score'])
    generic_list.append(line['Q2']['Generic']['Score'])
print('Hallucination:', np.sum(hallucinate_list)/20.)
print('Generic:', np.sum(generic_list)/20.)
print('Cooperativeness:', np.sum(coop_list)/20.)
print('Abstractiveness:', np.sum(abs_list)/20.)
print('Engagingness:', np.sum(engage_list)/20.)