from pathlib import Path
from typing import Any, List, Optional

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader


class ExcelReader(BaseReader):
    def __init__(
        self,
        *args: Any,
        pandas_config: Optional[dict] = None,
        row_joiner: str = "\n",
        col_joiner: str = " ",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ExcelReader with optional pandas configuration and joiners.

        Args:
            pandas_config (dict, optional): Configuration for pandas read_excel function.
            row_joiner (str, optional): Joiner for rows in the output document.
            col_joiner (str, optional): Joiner for columns in the output document.
        """
        super().__init__(*args, **kwargs)
        self._pandas_config = pandas_config or {}
        self._row_joiner = row_joiner if row_joiner else "\n"
        self._col_joiner = col_joiner if col_joiner else " "

    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
        sheet_name: str | int | List[str | int] | None = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed. Please install it with 'pip install pandas'."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in [".xlsx", ".xls"]:
            raise ValueError(f"File must be an Excel file (.xlsx or .xls): {file_path}")

        # Read the Excel file
        excel_file = pd.ExcelFile(file_path)
        file_name = file_path.name

        if not extra_info:
            extra_info = {}

        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split
        extra_info["file_size"] = file_path.stat().st_size
        extra_info["file_type"] = "excel"
        extra_info["sheet_names"] = excel_file.sheet_names

        # Handle sheet selection
        if sheet_name is None:
            sheets_to_process = excel_file.sheet_names
        elif isinstance(sheet_name, (str, int)):
            sheets_to_process = [sheet_name]
        else:
            sheets_to_process = sheet_name

        df_sheets = []
        sheet_infos = []
        for sheet in sheets_to_process:
            try:
                temp = []
                df = pd.read_excel(excel_file, sheet_name=sheet)
                # Add sheet-specific metadata
                sheet_extra_info = extra_info.copy()
                sheet_extra_info["source"] = f"{file_name}: Sheet {sheet}"
                sheet_extra_info["num_rows"] = len(df)
                sheet_extra_info["num_cols"] = len(df.columns)
                sheet_extra_info["columns"] = list(df.columns)

                df = df.dropna(axis=0, how="all")
                df = df.fillna("")

                temp.extend(df.values.astype(str).tolist())
                df_sheets.append(temp)
                sheet_infos.append(sheet_extra_info)

            except Exception as e:
                print(f"Warning: Could not process sheet '{sheet}': {str(e)}")
                continue

        outputs = [
            Document(
                document=self._row_joiner.join(
                    [self._col_joiner.join(row) for row in text_list]
                ).encode(encoding),
                metadata=sheet_infos[i],
            )
            for i, text_list in enumerate(df_sheets)
        ]
        return outputs
