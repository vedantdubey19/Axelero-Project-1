"""
Image extraction service.

Extracts every embedded raster image from a PDF page using PyMuPDF
and saves each one to output/images/<file_id>/.
"""

from pathlib import Path

import fitz

try:
    from pdf_parser_module.app.core.logger import logger
    from pdf_parser_module.app.models.schemas import ImageData
    from pdf_parser_module.app.utils.image_utils import save_image_bytes
except ImportError:
    from app.core.logger import logger
    from app.models.schemas import ImageData
    from app.utils.image_utils import save_image_bytes


def extract_images_from_page(
    document: fitz.Document,
    page: fitz.Page,
    page_number: int,
    output_folder: Path,
) -> list[ImageData]:
    """
    Extract all embedded images from a single page.

    Args:
        document: the parent PyMuPDF Document (needed to resolve image
            references, since a page only stores a reference to the
            image, not the raw bytes itself).
        page: the page to extract images from.
        page_number: 1-indexed page number, used in output filenames.
        output_folder: directory this file's images should be saved to.

    Returns:
        A list of ImageData objects describing each saved image.
        Images that fail to extract are skipped with a warning logged,
        rather than aborting the whole page.
    """
    extracted_images: list[ImageData] = []
    image_list = page.get_images(full=True)

    for index, image_info in enumerate(image_list):
        xref = image_info[0]  # internal PDF cross-reference number

        try:
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            extension = base_image.get("ext", "png")

            filename = f"page_{page_number}_image_{index}.{extension}"
            destination = output_folder / filename

            width, height = save_image_bytes(image_bytes, destination)

            extracted_images.append(
                ImageData(
                    image_index=index,
                    image_path=str(destination),
                    width=width,
                    height=height,
                )
            )

        except Exception as error:
            # A single bad image should not stop extraction of the
            # rest of the page, so we log and continue.
            logger.warning(
                f"Skipped image {index} on page {page_number}: {error}"
            )
            continue

    if extracted_images:
        logger.info(
            f"Extracted {len(extracted_images)} image(s) from page {page_number}"
        )

    return extracted_images
