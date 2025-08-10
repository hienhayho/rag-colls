import os
import json
import argparse
def main():
    parser = argparse.ArgumentParser(description="Evaluate page_to_markdown task")
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder",
    )
    parser.add_argument("--show_page_result", action="store_true", help="Whether to show the markdown of each page")
    args = parser.parse_args()

    src_dir = os.path.join(args.workspace, "results")
    tgt_dir = os.path.join(args.workspace, "markdowns")
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    for jsonl_file in os.listdir(src_dir):
        if jsonl_file.endswith(".jsonl"):
            with open(os.path.join(src_dir, jsonl_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    markdown_text = data['document_text']
                    file_name = os.path.basename(data['orig_path']).split(".")[0]
                    file_dir = os.path.join(tgt_dir, file_name)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    with open(os.path.join(file_dir, file_name+".md"), "w") as f:
                        f.write(markdown_text)
                    if args.show_page_result:
                        page_texts = data["page_texts"]
                        for page_num in page_texts.keys():
                            page_text = page_texts[page_num]
                            with open(os.path.join(file_dir, file_name+"_"+str(page_num)+".md"), "w") as f:
                                f.write(page_text)

if __name__ == "__main__":
    main()
