#!/bin/bash

# page_to_markdown task
python -m ocrflux.pipeline ./eval_page_to_markdown_result --task pdf2markdown --data /data/OCRFlux-bench-single/pdfs/*.pdf --model /data/OCRFlux-7B

python -m eval.eval_page_to_markdown ./eval_page_to_markdown_result --gt_file /data/OCRFlux-bench-single/data.jsonl

# element_merge_detect task
python -m eval.gen_element_merge_detect_data /data/OCRFlux-bench-cross

python -m ocrflux.pipeline ./eval_element_merge_detect_result --task merge_pages --data /data/OCRFlux-bench-cross/jsons/*.json --model /data/OCRFlux-7B

python -m eval.eval_element_merge_detect ./eval_element_merge_detect_result --gt_file /data/OCRFlux-bench-cross/data.jsonl

# table_to_html task
python -m ocrflux.pipeline ./eval_table_to_html_result --task pdf2markdown --data /data/OCRFlux-pubtabnet-single/images/*.png --model /data/OCRFlux-7B

python -m eval.eval_table_to_html ./eval_table_to_html_result --gt_file /data/OCRFlux-pubtabnet-single/data.jsonl

# html_table_merge task
python -m eval.gen_html_table_merge_data /data/OCRFlux-pubtabnet-cross

python -m ocrflux.pipeline ./eval_html_table_merge_result --task merge_tables --data /data/OCRFlux-pubtabnet-cross/jsons/*.json --model /data/OCRFlux-7B

python -m eval.eval_html_table_merge ./eval_html_table_merge_result --gt_file /data/OCRFlux-pubtabnet-cross/data.jsonl
