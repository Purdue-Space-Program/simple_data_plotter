import csv
from pathlib import Path


def truncate_csv_after_maximum_row_count(
    input_csv_file_path: str,
    output_csv_file_path: str,
    maximum_number_of_rows_to_keep: int = 100
) -> None:
    """
    Copies a CSV file while keeping only the first N rows.

    Parameters
    ----------
    input_csv_file_path : str
        Path to the original CSV file.
    output_csv_file_path : str
        Path where the truncated CSV file will be written.
    maximum_number_of_rows_to_keep : int
        Number of rows to keep (excluding header).
    """

    input_csv_path_object = Path(input_csv_file_path)
    output_csv_path_object = Path(output_csv_file_path)

    with input_csv_path_object.open(mode="r", newline="", encoding="utf-8") as input_file_handle, \
         output_csv_path_object.open(mode="w", newline="", encoding="utf-8") as output_file_handle:

        csv_reader_object = csv.reader(input_file_handle)
        csv_writer_object = csv.writer(output_file_handle)

        for current_row_index, current_row_data in enumerate(csv_reader_object):
            if current_row_index <= maximum_number_of_rows_to_keep:
                csv_writer_object.writerow(current_row_data)
            else:
                break


truncate_csv_after_maximum_row_count(
    input_csv_file_path=f"11-22-2025-methane_fill_csv.csv",
    output_csv_file_path="test_data_truncated.csv",
    maximum_number_of_rows_to_keep=100
)
