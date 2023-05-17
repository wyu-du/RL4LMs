import json

with open('data/multidoc2dial/multidoc2dial_dial_train.json', 'r') as f:
    all_dialogs = json.load(f)['dial_data']
with open('data/mdd_conv/dprft_train_10.json', 'r') as f:
    ori_data = json.load(f)
with open('data/multidoc2dial/multidoc2dial_doc.json', 'r') as f:
    all_docs = json.load(f)['doc_data']

def search_dial(dial_id):
    for domain in all_dialogs.keys():
        dialogs = all_dialogs[domain]
        for dialog in dialogs:
            if dialog['dial_id'] == dial_id:
                return dialog

def search_doc_span(doc_id, sp_id):
    for domain in all_docs.keys():
        docs = all_docs[domain]
        if doc_id in docs.keys():
            span_text = docs[doc_id]['spans'][sp_id]['text_sp']
            return span_text
        
def insert_special_token(string, substring):
    if substring in string:
        modified_string = string.replace(substring, "<BSP>" + substring + "<ESP>")
        return modified_string
    else:
        return string
    
outs = []
for line in ori_data:
    dialog = search_dial(line['conv_id'])
    turn_id = int(line['turn_id'])
    dialog_turn = dialog['turns'][turn_id]
    span_list = []
    for ref in dialog_turn['references']:
        sp_text = search_doc_span(ref['doc_id'], ref['id_sp'])
        span_list.append(sp_text.strip())
    # only use gold passage for training
    flag = True
    for n in range(100):
        if line['ctxs'][n]['has_answer'] is False: continue
        passage = line['ctxs'][n]['text']
        flag = False
        break
    if flag:
        passage = line['ctxs'][0]['text']
    new_line = line.copy()
    for sp in span_list:
        passage = insert_special_token(passage, sp)
    new_line['ctxs'] = passage
    new_line['sp_text'] = ' '.join(span_list)
    outs.append(new_line)
print(f'Total data: {len(ori_data)}')
print(f'New data: {len(outs)}')

with open('data/mdd_span_ann/dprft_train_10.json', 'w') as f:
    json.dump(outs, f, indent=4)
