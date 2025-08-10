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

    jsonl_file = os.path.join(args.workspace, 'data.jsonl')
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            json_name = data['image_name'].split('.')[0] + '.json'

            json_path = os.path.join(json_dir, json_name)
            data = {
                "table_1": data['table_fragment_1'],
                "table_2": data['table_fragment_2'],
            }
            with open(json_path, 'w') as f:
                json.dump(data, f)

if __name__ == "__main__":
    main()
