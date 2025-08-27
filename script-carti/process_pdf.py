# process_pdf.py
import os
import yaml
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(book_output_path):
    os.makedirs(os.path.join(book_output_path, 'raw_pages'), exist_ok=True)
    os.makedirs(os.path.join(book_output_path, 'split_pages'), exist_ok=True)


def pdf_to_images(pdf_path, raw_pages_dir, dpi=300):
    """Convert PDF to images with a visible progress bar.
    Uses per-page conversion so progress updates live.
    """
    info = pdfinfo_from_path(pdf_path)
    total_pages = int(info.get('Pages', 0))

    image_paths = []
    pbar = tqdm(total=total_pages, desc='[1/2] PDF → images', unit='page')

    # Convert page-by-page so the progress bar updates immediately
    for page in range(1, total_pages + 1):
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=page, last_page=page)
        page_img = pages[0]
        img_path = os.path.join(raw_pages_dir, f'scan_{page:03}.jpg')
        page_img.save(img_path, 'JPEG')
        image_paths.append((page, img_path))
        pbar.update(1)
    pbar.close()
    return image_paths


def split_image(image_path, split_type, output_dir, page_counter):
    img = Image.open(image_path)
    width, height = img.size

    if split_type == 'single':
        output_path = os.path.join(output_dir, f'page_{page_counter:03}.jpg')
        img.save(output_path)
        return [output_path]

    elif split_type == 'double':
        left = img.crop((0, 0, width // 2, height))
        right = img.crop((width // 2, 0, width, height))

        left_path = os.path.join(output_dir, f'page_{page_counter:03}.jpg')
        right_path = os.path.join(output_dir, f'page_{page_counter+1:03}.jpg')

        left.save(left_path)
        right.save(right_path)

        return [left_path, right_path]

    else:
        raise ValueError(f"Unsupported split type: {split_type}")


def get_split_type_for_page(split_config, page_num):
    for rule in split_config:
        if '-' in str(rule['page']):
            start, end = map(int, str(rule['page']).split('-'))
            if start <= page_num <= end:
                return rule['type']
        else:
            if int(rule['page']) == page_num:
                return rule['type']
    return 'double'  # fallback default


def slugify(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_').lower()


def main():
    config = load_config('configs/families_crisis.yaml')
    book_title    = config['book_title']
    pdf_file      = config['pdf_file']
    split_config  = config['split_strategy']
    dpi           = config.get('dpi', 300)  # <— citește DPI din YAML

    book_output_path = os.path.join('output', slugify(book_title))
    ensure_dirs(book_output_path)

    raw_dir   = os.path.join(book_output_path, 'raw_pages')
    split_dir = os.path.join(book_output_path, 'split_pages')

    print(f"[INFO] Converting PDF to images at {dpi} DPI...")
    page_images = pdf_to_images(pdf_file, raw_dir, dpi=dpi)  # <— folosește dpi din config

    print("[INFO] Splitting pages...")
    page_counter = 1
    pbar = tqdm(total=len(page_images), desc='[2/2] Split images', unit='scan')
    for page_num, image_path in page_images:
        split_type = get_split_type_for_page(split_config, page_num)
        output_paths = split_image(image_path, split_type, split_dir, page_counter)
        page_counter += len(output_paths)
        pbar.update(1)
    pbar.close()

    print("[DONE] Pages split and saved to:", split_dir)


if __name__ == '__main__':
    main()
