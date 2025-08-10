import os
import json
import argparse
import nltk
from tqdm import tqdm
from eval.parallel import parallel_process


def evaluate(pred, gt):
    pred = sorted(pred, key=lambda x: (x[0], x[1]))
    gt = sorted(gt, key=lambda x: (x[0], x[1]))
    if pred == gt:
        return 1
    else:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate element_merge_detect task")
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
    root_dir = os.path.join(args.workspace, "results")
    for jsonl_file in os.listdir(root_dir):
        if jsonl_file.endswith(".jsonl"):
            with open(os.path.join(root_dir, jsonl_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    pred_data[os.path.basename(data['orig_path'])] = data['merge_pairs']

    filename_list_en = []
    filename_list_zh = []
    gt_data = {}
    with open(args.gt_file, "r") as f:
        for line in f:
            data = json.loads(line)
            pdf_name_1 = data['pdf_name_1'].split(".")[0]
            pdf_name_2 = data['pdf_name_2'].split(".")[0]

            pdf_name,page_1 = pdf_name_1.split('_')
            pdf_name,page_2 = pdf_name_2.split('_')

            json_name = pdf_name + '_' + page_1 + '_' + page_2 + '.json'
            gt_data[json_name] = data['merging_idx_pairs']

            if data['language'] == 'en':
                filename_list_en.append(json_name)
            else:
                filename_list_zh.append(json_name)

    keys = list(gt_data.keys())
    if args.n_jobs == 1:
        scores = [evaluate(pred_data.get(filename, []), gt_data.get(filename, [])) for filename in tqdm(keys)]
    else:
        inputs = [{'pred': pred_data.get(filename, []), 'gt': gt_data.get(filename, [])} for filename in keys]
        scores = parallel_process(inputs, evaluate, use_kwargs=True, n_jobs=args.n_jobs, front_num=1)

    tp_en = 0
    tn_en = 0
    fp_en = 0
    fn_en = 0
    tp_zh = 0
    tn_zh = 0
    fp_zh = 0
    fn_zh = 0
    score_en = 0
    score_zh = 0
    num_en = 0
    num_zh = 0
    for filename, score in zip(keys, scores):
        print(filename)
        print(score)
        print()
        pred_label = pred_data[filename]
        if filename in filename_list_en:
            if pred_label == []:
                if score == 1:
                    tn_en += 1
                else:
                    fn_en += 1
            else:
                if score == 1:
                    tp_en += 1
                else:
                    fp_en += 1
            score_en += score
            num_en += 1

        elif filename in filename_list_zh:
            if pred_label == []:
                if score == 1:
                    tn_zh += 1
                else:
                    fn_zh += 1
            else:
                if score == 1:
                    tp_zh += 1
                else:
                    fp_zh += 1
            score_zh += score
            num_zh += 1

    precision_en = tp_en / (tp_en + fp_en)
    recall_en = tp_en / (tp_en + fn_en)
    f1_en = 2*precision_en*recall_en / (precision_en+recall_en)
    acc_en = score_en / num_en

    precision_zh = tp_zh / (tp_zh + fp_zh)
    recall_zh = tp_zh / (tp_zh + fn_zh)
    f1_zh = 2*precision_zh*recall_zh / (precision_zh+recall_zh)
    acc_zh = score_zh / num_zh

    tp = tp_en + tp_zh
    fp = fp_en + fp_zh
    fn = fn_en + fn_zh
    score = score_en + score_zh
    num = num_en + num_zh

    precision = tp / (tp + fp)
    recall =  tp / (tp + fn)
    f1 = 2*precision*recall / (precision+recall)
    acc = score / num

    print(f"EN: {precision_en} / {recall_en} / {f1_en} / {acc_en}")
    print(f"ZH: {precision_zh} / {recall_zh} / {f1_zh} / {acc_zh}")
    print(f"ALL: {precision} / {recall} / {f1} / {acc}")

if __name__ == "__main__":
    main()
