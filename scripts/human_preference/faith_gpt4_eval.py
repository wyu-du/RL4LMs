import json
import time
import openai

with open('scripts/openai_api_key', 'r') as f:
  openai.api_key = f.read().strip()

question_list, knowledge_list = [], []
with open('outputs/mdd_refs.jsonl', 'r') as f:
    lines = f.read().strip().split('\n')
    for line in lines:
        knowledge_text = json.loads(line)['knowledge_text']
        knowledge_text = ' '.join(knowledge_text.split(' ')[:200])
        knowledge_list.append(knowledge_text)
        question_list.append(json.loads(line)['prompt_or_input_text'])
response_list = []
with open('outputs/mdd_alpha_025.json', 'r') as f:
    lines = json.load(f)
    # lines = f.read().strip().split('\n')
    for line in lines:
        response_list.append(line['generated_text'])
        # response_list.append(json.loads(line)['response'])

incontext_example = """
You are CompareGPT, a machine to verify the groundedness of predictions. Answer with only yes/no. 
You are given a question, the corresponding evidence and a prediction from a model. 
Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction in present in the evidence or can be inferred from the evidence. 
You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence. 
"""

pred_list = []
# fout = open('outputs/mdd_alpha_human_gpt4_eval.jsonl', 'w')
for i in range(len(response_list)):
  if i >= 100: break
  question = question_list[i].split('[SEP]')[0].strip()
  question = question.split('question:')[-1].strip()
  knowledge = knowledge_list[i].strip()
  response = response_list[i].strip()
  output = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
          {"role": "system", "content": incontext_example},
          {"role": "user", "content": f"Question: {question}\nPrediction: {response}\nEvidence: {knowledge}\nCompareGPT response:"}
      ],
    temperature=0.9,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
  )
  time.sleep(5)
  answer = output['choices'][0]['message']['content'].lower()
  print(answer)
  if 'no' in answer:
    pred_list.append(0)
    tmp = {'question': question, 'knowledge': knowledge, 'response': response, 'gpt4_eval': 0}
  else:
    pred_list.append(1)
    tmp = {'question': question, 'knowledge': knowledge, 'response': response, 'gpt4_eval': 1}
#   fout.write(json.dumps(tmp)+'\n')
score = sum(pred_list)/float(len(pred_list))
print(score * 100)