import json
import time
import openai
from datasets import load_metric
sacrebleu = load_metric("sacrebleu")
bertscore = load_metric("bertscore")

with open('scripts/openai_api_key', 'r') as f:
  openai.api_key = f.read().strip()

with open('data/human_preference/t5_faithdial.json', 'r') as f:
  lines = json.load(f)

# incontext_example = """
# The following is a conversation with an AI assistant. 
# The assistant is providing answer based on a knowledge passage.
# The following is an example.\n\n
# Human: Hello, I forgot to update my address, can you help me with that?\n
# Knowledge passage: Top 5 DMV Mistakes and How to Avoid Them //   1. Forgetting to Update Address   By statute ,  you must report a change of address to DMV within ten days of moving.  That is the case for the address associated with your license, as well as all the addresses associated with each registered vehicle, which may differ.  It is not sufficient to only: write your new address on the back of your old license; tell the United States Postal Service; or inform the police officer writing you a ticket.  If you fail to keep your address current ,  you will miss a suspension order and may be charged with operating an unregistered vehicle and/or aggravated unlicensed operation, both misdemeanors.  This really happens ,  but the good news is this is a problem that is easily avoidable.  Learn more about how to change the address on your license and registrations [1]\n
# AI: hi, you have to report any change of address to DMV within 10 days after moving. You should do this both for the address associated with your license and all the addresses associated with all your vehicles.\n
# """

incontext_example = """
The following is a conversation with an AI assistant. 
The assistant is providing answer based on a knowledge passage.
The following is an example.\n\n
Human: I recently discovered rap music and I'm so intrigued by it! Do you listen to rap music?\n
Knowledge passage: Rapping (or rhyming, spitting, emceeing, MCing) is a musical form of vocal delivery that incorporates ''rhyme, rhythmic speech, and street vernacular'', which is performed or chanted in a variety of ways, usually over a backbeat or musical accompaniment.\n
AI: No, I'm a bot and can't hear. I know that it's a form of music that involves chanting and rhythmic speech.\n
"""


outs = []
for line in lines:
  question = line['question'].split('[SEP]')[0].strip()
  knowledge = line['ctxs'].strip()
  sp_text = line['sp_text']
  ref_text = line['answers'][0]
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"{incontext_example}Human: {question}\nKnowledge passage: {knowledge}\nAI:",
    temperature=0.9,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:", "Knowledge passage:"]
  )
  time.sleep(10)
  answer = response['choices'][0]['text']
  metric_results = bertscore.compute(predictions=[answer], references=[sp_text],
                                     lang='en', device="cuda:0")
  bert_know_f1 = metric_results["f1"][0]
  gpt_sacrebleu = sacrebleu.compute(predictions=[answer], references=[[ref_text]])
  gpt_sacrebleu = gpt_sacrebleu['score']/100
  tmp = line
  tmp['gpt_generated_text'] = answer
  tmp['gpt_sacre_bleu'] = gpt_sacrebleu
  tmp['gpt_bert_know_f1'] = bert_know_f1
  outs.append(tmp)
  print(tmp)

with open('data/human_preference/gpt_faithdial.json', 'w') as f:
    json.dump(outs, f, indent=4)
print(len(outs))
