import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate element_merge_detect task")
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder",
    )
    args = parser.parse_args()

    json_dir = os.path.join(args.workspace, 'jsons')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    jsonl_file = os.path.join(args.workspace, "data.jsonl")
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            pdf_name_1 = data['pdf_name_1'].split(".")[0]
            pdf_name_2 = data['pdf_name_2'].split(".")[0]

            pdf_name,page_1 = pdf_name_1.split('_')
            pdf_name,page_2 = pdf_name_2.split('_')

            json_name = os.path.join(json_dir, pdf_name + '_' + page_1 + '_' + page_2 + '.json')
            data = {
                "page_1": "\n\n".join(data['md_elem_list_1']),
                "page_2": "\n\n".join(data['md_elem_list_2']),
            }
            with open(json_name, 'w') as f:
                json.dump(data, f)

if __name__ == "__main__":
    main()
