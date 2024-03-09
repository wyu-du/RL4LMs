import json

context_list, question_list, knowledge_list = [], [], []
# with open('data/mdd_span_only/dprft_test.json', 'r') as f:
#     span_only_refs = json.load(f)
with open('outputs/fd_refs.jsonl', 'r') as f:
    lines = f.read().strip().split('\n')
    for i, line in enumerate(lines):
        knowledge_text = json.loads(line)['knowledge_text']
        # knowledge_text = span_only_refs[i]['sp_text']
        knowledge_list.append(knowledge_text)
        prompt_or_input_text = json.loads(line)['prompt_or_input_text']
        question, context = prompt_or_input_text.split('[SEP]')
        question_list.append(question.replace('question:', 'user:'))
        cur_list = []
        if 'context: ' in context:
            context = context.split('context: ')[0]
        if '||' in context:
            utts = context.split('||')
            for utt in utts[::-1]:
                cur_list.append(utt.strip())
        else:
            cur_list.append(context.strip())
        context_list.append(cur_list)

response_list = []
with open('outputs/fd_alpha_human_gpt4_eval.jsonl', 'r') as f:
    # lines = json.load(f)
    lines = f.read().strip().split('\n')
    for line in lines:
        # response_list.append(line['generated_text'])
        response_list.append(json.loads(line)['response'])

outs = []
j = 0
for i in range(100):
    if j == 20: break
    if len(context_list[i]) > 1 and len(context_list[i]) < 5: 
        tmp = {
            "Dialogue": context_list[i] + [question_list[i]],
            "Knowledge Text": knowledge_list[i],
            "Model Response": "agent: "+response_list[i],
            "Q1": {
                "Question": "If the model response is grounded on the given knowledge text, please score the response with the following three qualities (in 1-4 Likert score, 1 means very bad, 4 means very good).",
                "Cooperativeness": {
                    "Definition": "The response is coherent with the previous turn and does not try to mislead the speaker or act unhelpfully.",
                    "Score": 0
                },
                "Engagingness": {
                    "Definition": "The response involves engaging the speaker by prompting further replies and moving the conversation forward.",
                    "Score": 0
                },
                "Abstractiveness": {
                    "Definition": "The response reuses information from the source knowledge in a novel way.",
                    "Score": 0
                }
            },
            "Q2": {
                "Question": "If the model response is NOT grounded on the given knowledge text, please mark the response as hallucination or generic (0 means no, 1 means yes).",
                "Hallucination": {
                    "Definition": "The response contains some information unsupported by the knowledge text.",
                    "Score": 0
                },
                "Generic": {
                    "Definition": "The response is simple and uninformative, such as “I don't know”, “That's nice”.",
                    "Score": 0
                }
            }
        }
        outs.append(tmp)
        j += 1
print(len(outs))
with open('outputs/human_eval_fd_model1.json', 'w') as f:
    json.dump(outs, f, indent=4)