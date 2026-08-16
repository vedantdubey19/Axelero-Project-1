"""Table extraction service using pdfplumber."""

from collections import defaultdict
from pathlib import Path
import csv

import pdfplumber

from app.core.logger import logger
from app.models.schemas import TableData


def extract_all_tables(
    file_path: Path,
    total_pages: int,
    output_folder: Path,
) -> dict[int, list[TableData]]:
    """Extract tables from every PDF page using pdfplumber.

    Returns a mapping of 1-indexed page number to extracted tables.
    Each table is saved as a CSV file in ``output_folder``.
    """
    results: dict[int, list[TableData]] = {
        page_number: [] for page_number in range(1, total_pages + 1)
    }

    if total_pages == 0:
        return results

    try:
        with pdfplumber.open(file_path) as pdf:
            output_folder.mkdir(parents=True, exist_ok=True)

            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception as error:
                    logger.warning(
                        f"Table extraction issue on page {page_number} of "
                        f"{file_path.name}: {error}"
                    )
                    continue

                for table_index, table in enumerate(tables):
                    if not table:
                        continue

                    rows = [
                        ["" if cell is None else str(cell).strip() for cell in row]
                        for row in table
                    ]
                    rows = [row for row in rows if any(cell for cell in row)]
                    if not rows:
                        continue

                    filename = f"page_{page_number}_table_{table_index}.csv"
                    destination = output_folder / filename

                    try:
                        with destination.open("w", newline="", encoding="utf-8") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerows(rows)

                        column_count = max(len(row) for row in rows)
                        results[page_number].append(
                            TableData(
                                table_index=table_index,
                                csv_path=str(destination),
                                rows=len(rows),
                                columns=column_count,
                            )
                        )
                    except OSError as error:
                        logger.warning(
                            f"Failed to save table {table_index} on page "
                            f"{page_number}: {error}"
                        )

    except Exception as error:
        logger.warning(f"Could not open {file_path.name} for table extraction: {error}")

    tables_found = sum(len(page_tables) for page_tables in results.values())
    if tables_found:
        logger.info(f"Extracted {tables_found} table(s) from {file_path.name}")

    return results
