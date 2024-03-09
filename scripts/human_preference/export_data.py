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

def parse_to_excel(lines, fpath='data/human_preference/sampled_data3.xlsx'):
    workbook = xlsxwriter.Workbook(fpath)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 12)
    worksheet.set_column(1, 2, 50)
    worksheet.set_column(3, 3, 100)
    worksheet.set_column(4, 5, 50)
    worksheet.set_column(6, 14, 20)
    green = workbook.add_format({'color': 'green'})
    bold = workbook.add_format({'bold': True})
    merge = workbook.add_format({'text_wrap': True, 'align': 'left', 'valign': 'vcenter'})

    worksheet.write('A1', 'id')
    worksheet.write('B1', 'history')
    worksheet.write('C1', 'question')
    worksheet.write('D1', 'knowledge_passage')
    worksheet.write('E1', 't5_ft (t5)')
    worksheet.write('F1', 'gpt_3.5_turbo (chatgpt)')
    worksheet.write('G1', 't5_sacre_bleu')
    worksheet.write('H1', 't5_token_know_f1')
    worksheet.write('I1', 'chatgpt_sacre_bleu')
    worksheet.write('J1', 'chatgpt_token_know_f1')
    worksheet.write('K1', 'label')
    worksheet.set_row(0, None, bold)
    worksheet.freeze_panes(1, 0)
    
    m = 2
    for i in range(len(lines)):
        history = ' '.join(lines[i]['question'].split('[SEP]')[-1].split()[:150])
        question = lines[i]['question'].split('[SEP]')[0]
        text = annotate_text(lines[i]['ctxs'], lines[i]['sp_text'], green)
        
        worksheet.write(f'A{m}', m-1, merge)
        worksheet.write(f'B{m}', history, merge)
        worksheet.write(f'C{m}', question, merge)
        worksheet.write_rich_string(f'D{m}', *text, merge)
        worksheet.write(f'E{m}', lines[i]['t5_ft_generated_text'], merge)
        worksheet.write(f'F{m}', lines[i]['chatgpt_generated_text'], merge)
        worksheet.write(f'G{m}', lines[i]['t5_ft_sacre_bleu'])
        worksheet.write(f'H{m}', lines[i]['t5_token_know_f1'])
        worksheet.write(f'I{m}', lines[i]['chatgpt_sacre_bleu'])
        worksheet.write(f'J{m}', lines[i]['chatgpt_token_know_f1'])
        m += 1
    workbook.close()

with open('data/human_preference/chatgpt_multidoc2dial_2024.json', 'r') as f:
    lines = json.load(f)

parse_to_excel(lines, fpath='data/human_preference/sampled_multidoc2dial_2024.xlsx')