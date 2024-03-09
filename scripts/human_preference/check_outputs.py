import json
# from rl4lms.envs.text_generation.reward import f1_score

# with open(f'ckpts/t5-base-fd-supervised/epoch_10_train_split_predictions.json') as f:
#     lines = json.load(f)
# with open(f'rl4lm_exps/t5-fd-supervised-ppo-full-bleu_no_kl/epoch_199_train_split_predictions.json') as f:
#     lines2 = json.load(f)
# for data_idx in range(10):
#     knowledge = lines[data_idx]['prompt_text'].split('context: ')[-1]
#     ref_text = lines[data_idx]['ref_text'].replace('<START-1>', '').replace('<END-1>', '')
#     generated_text = lines[data_idx]['generated_text']
#     generated_text2 = lines2[data_idx]['generated_text']
#     # ref_f1 = lines[data_idx]['lexical/reference_f1']
#     print(f'=========== {data_idx} ===========')
#     print('knowledge_text:', knowledge)
#     print('reference_text:', ref_text)
#     print('sft_generated_text:', generated_text)
#     print('ppo_generated_text:', generated_text2)
#     # print('ref_token_f1:', ref_f1)

data_idx = 11
with open(f'rl4lm_exps/t5-fd-supervised-ppo-full-token_ref_f1_new/epoch_0_val_split_predictions.json') as f:
    lines = json.load(f)
knowledge = lines[data_idx]['prompt_text'].split('context: ')[-1]
ref_text = lines[data_idx]['ref_text'].replace('<START-1>', '').replace('<END-1>', '')
generated_text = lines[data_idx]['generated_text']
ref_f1 = lines[data_idx]['lexical/reference_f1']
print(f'=========== 0 ===========')
print('knowledge_text:', knowledge)
print('ref_text:', ref_text)
print('generated_text:', generated_text)
print('ref_token_f1:', ref_f1)

for i in range(99, 1800, 100):
    with open(f'rl4lm_exps/t5-fd-supervised-ppo-full-token_ref_f1_new/epoch_{i}_val_split_predictions.json') as f:
        lines = json.load(f)
    knowledge = lines[data_idx]['prompt_text'].split('context: ')[-1]
    ref_text = lines[data_idx]['ref_text'].replace('<START-1>', '').replace('<END-1>', '')
    generated_text = lines[data_idx]['generated_text']
    ref_f1 = lines[data_idx]['lexical/reference_f1']
    print(f'=========== {i} ===========')
    print('knowledge_text:', knowledge)
    print('ref_text:', ref_text)
    print('generated_text:', generated_text)
    print('ref_token_f1:', ref_f1)