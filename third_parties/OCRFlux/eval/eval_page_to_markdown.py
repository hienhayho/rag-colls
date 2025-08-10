import os
import json
import argparse
import nltk
from tqdm import tqdm
from eval.parallel import parallel_process


def evaluate(pred, gt):
    edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return 1.0 - edit_dist


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
    root_dir = os.path.join(args.workspace, "results")
    for jsonl_file in os.listdir(root_dir):
        if jsonl_file.endswith(".jsonl"):
            with open(os.path.join(root_dir, jsonl_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    pred_data[os.path.basename(data['orig_path'])] = data['document_text']

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
        print(filename)
        print(score)
        print()
        if filename in filename_list_en:
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
