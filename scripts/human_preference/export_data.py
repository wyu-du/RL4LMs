import xlsxwriter
import json

def find_substring_indices(string, substring):
    if substring not in string:
        return -1, -1
    else:
        start_index = string.index(substring)
        end_index = start_index + len(substring) - 1
        return start_index, end_index

def annotate_text(passage, span, green):
    sid, eid = find_substring_indices(passage, span)
    texts = [passage[:sid], green, span, passage[eid:]]
    return texts

def parse_to_excel(lines1, lines2, fpath='data/human_preference/sampled_data3.xlsx'):
    workbook = xlsxwriter.Workbook(fpath)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 12)
    worksheet.set_column(1, 2, 50)
    worksheet.set_column(3, 3, 100)
    worksheet.set_column(4, 6, 50)
    worksheet.set_column(7, 14, 20)
    green = workbook.add_format({'color': 'green'})
    bold = workbook.add_format({'bold': True})
    merge = workbook.add_format({'text_wrap': True, 'align': 'left', 'valign': 'vcenter'})

    worksheet.write('A1', 'id')
    worksheet.write('B1', 'history')
    worksheet.write('C1', 'question')
    worksheet.write('D1', 'knowledge_passage')
    worksheet.write('E1', 't5_ft (t5)')
    worksheet.write('F1', 'text_davinci_003 (gpt3.5)')
    worksheet.write('G1', 'gpt_3.5_turbo (chatgpt)')
    worksheet.write('H1', 't5_sacre_bleu')
    worksheet.write('I1', 't5_bert_know_f1')
    worksheet.write('J1', 'gpt3.5_sacre_bleu')
    worksheet.write('K1', 'gpt3.5_bert_know_f1')
    worksheet.write('L1', 'chatgpt_sacre_bleu')
    worksheet.write('M1', 'chatgpt_bert_know_f1')
    worksheet.set_row(0, None, bold)
    worksheet.freeze_panes(1, 0)
    
    m = 2
    for i in range(len(lines1)):
        history = ' '.join(lines1[i]['question'].split('[SEP]')[-1].split()[:150])
        question = lines1[i]['question'].split('[SEP]')[0]
        text = annotate_text(lines1[i]['ctxs'], lines1[i]['sp_text'], green)
        
        worksheet.write(f'A{m}', m-1, merge)
        worksheet.write(f'B{m}', history, merge)
        worksheet.write(f'C{m}', question, merge)
        worksheet.write_rich_string(f'D{m}', *text, merge)
        worksheet.write(f'E{m}', lines1[i]['t5_ft_generated_text'], merge)
        worksheet.write(f'F{m}', lines1[i]['gpt_generated_text'], merge)
        worksheet.write(f'G{m}', lines2[i]['chatgpt_generated_text'], merge)
        worksheet.write(f'H{m}', lines1[i]['t5_ft_sacre_bleu'])
        worksheet.write(f'I{m}', lines1[i]['t5_ft_bert_know_f1'])
        worksheet.write(f'J{m}', lines1[i]['gpt_sacre_bleu'])
        worksheet.write(f'K{m}', lines1[i]['gpt_bert_know_f1'])
        worksheet.write(f'L{m}', lines2[i]['chatgpt_sacre_bleu'])
        worksheet.write(f'M{m}', lines2[i]['chatgpt_bert_know_f1'])
        m += 1
    workbook.close()

with open('data/human_preference/chatgpt_faithdial.json', 'r') as f:
    lines2 = json.load(f)
with open('data/human_preference/gpt_faithdial.json', 'r') as f:
    lines1 = json.load(f)

parse_to_excel(lines1, lines2, fpath='data/human_preference/sampled_faithdial.xlsx')