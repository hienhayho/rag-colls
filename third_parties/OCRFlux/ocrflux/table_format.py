
from bs4 import BeautifulSoup
import re

def is_html_table(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.find('table') is not None

def table_matrix2html(matrix_table):
    soup = BeautifulSoup(matrix_table, 'html.parser')
    table = soup.find('table')
    rownum = 0
    colnum = 0
    cell_dict = {}
    rid = 0
    for tr in table.find_all('tr'):
        cid = 0
        for td in tr.find_all('td'):
            if td.find('l'):
                cell_dict[(rid, cid)] = '<l>'
            elif td.find('t'):
                cell_dict[(rid, cid)] = '<t>'
            elif td.find('lt'):
                cell_dict[(rid, cid)] = '<lt>'
            else:
                text = td.get_text(strip=True)
                cell_dict[(rid, cid)] = text
            cid += 1
        if colnum == 0:
            colnum = cid
        elif cid != colnum:
            raise Exception('colnum not match')
        rid += 1
    rownum = rid
    html_table = ['<table>']
    for rid in range(rownum):
        html_table.append('<tr>')
        for cid in range(colnum):
            if (rid, cid) not in cell_dict.keys():
                continue
            text = cell_dict[(rid, cid)]
            if text == '<l>' or text == '<t>' or text == '<lt>':
                raise Exception('cell not match')
            rowspan = 1
            colspan = 1
            for r in range(rid+1, rownum):
                if (r, cid) in cell_dict.keys() and cell_dict[(r, cid)] == '<t>':
                    rowspan += 1
                    del cell_dict[(r, cid)]
                else:
                    break
            for c in range(cid+1, colnum):
                if (rid, c) in cell_dict.keys() and cell_dict[(rid, c)] == '<l>':
                    colspan += 1
                    del cell_dict[(rid, c)]
                else:
                    break
            for r in range(rid+1, rid+rowspan):
                for c in range(cid+1, cid+colspan):
                    if cell_dict[(r, c)] != '<lt>':
                        raise Exception('cell not match')
                    del cell_dict[(r, c)]
            attr = ''
            if rowspan > 1:
                attr += ' rowspan="{}"'.format(rowspan)
            if colspan > 1:
                attr += ' colspan="{}"'.format(colspan)
            html_table.append("<td{}>{}</td>".format(attr, text))
        html_table.append('</tr>')
    html_table.append('</table>')
    return "".join(html_table)

def table_html2matrix(html_table):
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    rownum = len(table.find_all('tr'))
    colnum = 0
    tr = table.find_all('tr')[0]
    for td in tr.find_all('td'):
        colnum += td.get('colspan', 1)
    matrix = [[None for _ in range(colnum)] for _ in range(rownum)]

    rid = 0
    for tr in table.find_all('tr'):
        cid = 0
        for td in tr.find_all('td'):
            for c in range(cid, colnum):
                if matrix[rid][c] is None:
                    break
            cid = c
            rowspan = td.get('rowspan', 1)
            colspan = td.get('colspan', 1)
            cell_text = td.get_text(strip=True)
            for r in range(rid,rid+rowspan):
                if r >= rownum:
                    raise Exception('rownum not match')
                for c in range(cid,cid+colspan):
                    if c >= colnum:
                        raise Exception('colnum not match')
                    if matrix[r][c] is not None:
                        raise Exception('cell not match')
                    if r == rid and c == cid:
                        matrix[r][c] = cell_text
                    elif r == rid:
                        matrix[r][c] = '<l>'
                    elif c == cid:
                        matrix[r][c] = '<t>'
                    else:
                        matrix[r][c] = '<lt>'
            cid += colspan
        rid += 1

    matrix_table = ['<table>']
    for rid in range(rownum):
        matrix_table.append('<tr>')
        for cid in range(colnum):
            matrix_table.append('<td>')
            cell_text = matrix[rid][cid]
            matrix_table.append(cell_text)
            matrix_table.append('</td>')
        matrix_table.append('</tr>')
    matrix_table.append('</table>')
    return "".join(matrix_table)

trans_func = {
    "html2matrix": table_html2matrix,
    "matrix2html": table_matrix2html,
}

def trans_markdown_text(markdown_text,trans_type):
    if markdown_text == None:
        return None
    text_list = markdown_text.split('\n\n')
    for i,text in enumerate(text_list):
        if is_html_table(text):
            text_list[i] = trans_func[trans_type](text)
    return "\n\n".join(text_list)
