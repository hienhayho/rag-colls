import os
import json
import argparse
import distance
import markdown2
import re
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm
from eval.parallel import parallel_process
from bs4 import BeautifulSoup

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


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        pred.replace("<th>","<td>")
        pred.replace("</th>","</td>")
        pred = "<html>" + pred + "</html>"
        true = "<html>" + true + "</html>"
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        total_score_simple = 0
        num_simple = 0
        total_score_complex = 0
        num_complex = 0
        total_score = 0
        num_total = 0
        for filename,score in zip(samples, scores):
            print(filename)
            print(score)
            print('')
            if true_json[filename]['type'] == 'simple':
                total_score_simple += score
                num_simple += 1
            elif true_json[filename]['type'] == 'complex':
                total_score_complex += score
                num_complex += 1
            else:
                raise ValueError('Unknown type: %s' % true_json[filename]['type'])
            total_score += score
            num_total += 1
        if num_simple > 0:
            avg_score_simple = total_score_simple / num_simple
        else:
            avg_score_simple = 0
        if num_complex > 0:
            avg_score_complex = total_score_complex / num_complex
        else:
            avg_score_complex = 0
        avg_score = total_score / num_total
        print({'simple': (num_simple,avg_score_simple), 'complex': (num_complex,avg_score_complex), 'total': (num_total,avg_score)})

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
        pdf_name = file.split('.')[0] + ".png"
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


    gt_data = {}
    with open(args.gt_file, "r") as f:
        for line in f:
            data = json.loads(line)
            gt_data[data['image_name']] = {'html':data['gt_table'], 'type':data['type']}

    teds = TEDS(n_jobs=args.n_jobs, ignore_nodes=['b', 'thead', 'tbody'])
    teds.batch_evaluate(pred_data, gt_data)

if __name__ == "__main__":
    main()
