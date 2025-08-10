import os
import re
import json
import argparse
import nltk
import markdown2
from bs4 import BeautifulSoup
from tqdm import tqdm
from eval.parallel import parallel_process

def turn_header_to_h1(line):
    # 检查是否是以一个或多个 '#' 开头的标题行
    if line.lstrip().startswith('#'):
        # 去掉开头的 '#' 和其后的空格
        new_line = "# " + line.lstrip().lstrip('#').lstrip()
        return new_line
    else:
        return line

def replace_single_dollar(markdown_text):
    pattern = r'\$(.*?)\$'
    def replace_with_brackets(match):
        formula_content = match.group(1)  # 获取匹配到的公式内容
        return f'\\({formula_content}\\)'

    replaced_text = re.sub(pattern, replace_with_brackets, markdown_text, flags=re.DOTALL)

    return replaced_text


def replace_double_dollar(markdown_text):
    pattern = r'\$\$(.*?)\$\$'
    def replace_with_brackets(match):
        formula_content = match.group(1)
        return f'\\[{formula_content}\\]'
    replaced_text = re.sub(pattern, replace_with_brackets, markdown_text, flags=re.DOTALL)

    return replaced_text

def simplify_html_table(html_table):
    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(html_table, 'html.parser')

    # 找到 <table> 标签
    table = soup.find('table')
    if not table:
        raise ValueError("输入的 HTML 不包含有效的 <table> 标签")

    # 创建一个新的 <table> 标签
    new_table = BeautifulSoup('<table></table>', 'html.parser').table

    # 提取所有行（包括 <thead> 和 <tbody> 中的行）
    rows = table.find_all(['tr'], recursive=True)

    for row in rows:
        # 创建新的 <tr> 标签
        new_row = soup.new_tag('tr')

        # 处理每一行中的单元格
        cells = row.find_all(['th', 'td'])
        for cell in cells:
            # 将 <th> 替换为 <td>
            new_cell = soup.new_tag('td')
            if cell.has_attr('rowspan'):
                new_cell['rowspan'] = cell['rowspan']
            if cell.has_attr('colspan'):
                new_cell['colspan'] = cell['colspan']
            new_cell.string = cell.get_text(strip=True)  # 保留单元格内容
            new_row.append(new_cell)

        # 将新行添加到新表格中
        new_table.append(new_row)

    # 返回简化后的表格 HTML
    return str(new_table)

def evaluate(pred, gt):
    edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return 1.0- edit_dist


def main():
    parser = argparse.ArgumentParser(description="Evaluate page_to_markdown task")
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder",
    )
    parser.add_argument(
        "--gt_file",
        help="Ground truth file",
    )
    parser.add_argument("--n_jobs", type=int, default=40, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    pred_data = {}
    for file in os.listdir(args.workspace):
        file_path = os.path.join(args.workspace, file)
        pdf_name = file.split('.')[0] + ".pdf"
        with open(file_path, "r") as f:
            document_text = f.read()
            document_text = replace_single_dollar(replace_double_dollar(document_text))
            markdown_text_list = document_text.split("\n\n")
            new_markdown_text_list = []
            for text in markdown_text_list:
                text = text.strip()
                if (text.startswith("<watermark>") and text.endswith("</watermark>")) or (text.startswith("<img>") and text.endswith("</img>")) or (text.startswith("<page_number>") and text.endswith("</page_number>")) or (text.startswith("<signature>") and text.endswith("</signature>")):
                    continue
                else:
                    html_text = str(markdown2.markdown(text,extras=["tables"]))
                    html_text = html_text.strip()
                    if html_text.startswith("<table>") and html_text.endswith("</table>"):
                        html_table = simplify_html_table(html_text)
                        new_markdown_text_list.append(html_table)
                    else:
                        text = turn_header_to_h1(text)
                        new_markdown_text_list.append(text)

            pred_data[os.path.basename(pdf_name)] = "\n\n".join(new_markdown_text_list)

    filename_list_en = []
    filename_list_zh = []
    gt_data = {}
    with open(args.gt_file, "r") as f:
        for line in f:
            data = json.loads(line)
            markdown = data['markdown']
            pdf_name = data['pdf_name']
            gt_data[pdf_name] = markdown
            if data['language'] == 'en':
                filename_list_en.append(pdf_name)
            else:
                filename_list_zh.append(pdf_name)

    keys = list(gt_data.keys())
    if args.n_jobs == 1:
        scores = [evaluate(pred_data.get(filename, ''), gt_data.get(filename, '')) for filename in tqdm(keys)]
    else:
        inputs = [{'pred': pred_data.get(filename, ''), 'gt': gt_data.get(filename, '')} for filename in keys]
        scores = parallel_process(inputs, evaluate, use_kwargs=True, n_jobs=args.n_jobs, front_num=1)

    total_score_en = 0
    total_num_en = 0
    total_score_zh = 0
    total_num_zh = 0
    for filename, score in zip(keys, scores):
        if filename in filename_list_en:
            print(filename)
            print(score)
            print()
            total_score_en += score
            total_num_en += 1
        elif filename in filename_list_zh:
            total_score_zh += score
            total_num_zh += 1
    print(f"English: {total_score_en / total_num_en}")
    print(f"Chinese: {total_score_zh / total_num_zh}")
    print(f"Total: {sum(scores) / len(scores)}")

if __name__ == "__main__":
    main()
