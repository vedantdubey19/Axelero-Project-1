"""Tests for app/services/table_extractor.py"""

from pathlib import Path

try:
    from pdf_parser_module.app.services.table_extractor import extract_all_tables
except ImportError:
    from app.services.table_extractor import extract_all_tables


def test_extract_all_tables_with_no_tables_returns_empty_lists(
    sample_pdf_path: Path, tmp_path: Path
) -> None:
    """
    Our sample PDF fixture only contains plain text, no table
    structure, so extraction should gracefully return an empty list
    for the page rather than raising an error.
    """
    output_folder = tmp_path / "tables"

    results = extract_all_tables(sample_pdf_path, total_pages=1, output_folder=output_folder)

    assert results == {1: []}


def test_extract_all_tables_preserves_page_numbers_across_document(
    multi_page_pdf_path: Path, tmp_path: Path
) -> None:
    """
    Every page of the document should have an entry in the result,
    keyed by its correct 1-indexed page number, even when no tables
    are found anywhere in the document.
    """
    output_folder = tmp_path / "tables"

    results = extract_all_tables(multi_page_pdf_path, total_pages=20, output_folder=output_folder)

    assert set(results.keys()) == set(range(1, 21))
    assert all(tables == [] for tables in results.values())


def test_extract_all_tables_with_zero_pages_returns_empty_mapping(tmp_path: Path) -> None:
    output_folder = tmp_path / "tables"

    results = extract_all_tables(Path("unused.pdf"), total_pages=0, output_folder=output_folder)

    assert results == {}


def test_extract_all_tables_from_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent.parent / "sample_pdfs" / "table_test.pdf"
    output_folder = tmp_path / "tables"

    results = extract_all_tables(fixture, total_pages=1, output_folder=output_folder)

    assert len(results[1]) == 1
    table = results[1][0]
    assert table.rows == 6
    assert table.columns == 4
    assert Path(table.csv_path).exists()
