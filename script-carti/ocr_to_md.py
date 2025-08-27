# ocr_to_md.py
"""
OCR → Markdown pentru pagini scanate (după split).
- Preprocesare blândă (PIL grayscale + crop margini).
- OCR cu image_to_string; încearcă PSM [tess_psm, 3, 4] și alege cea mai bună.
- Evită junk: dacă textul e scurt sau conf. mică → inserează doar imaginea.
- Progress bar (tqdm).
- Interval opțional: --from-page / --to-page (1-based, inclusiv).
"""

from __future__ import annotations
import os, re, glob, csv, yaml
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pytesseract import image_to_string, image_to_data, Output

# ---------- Utilitare bază ----------

def load_config(path: str):
    """Citește YAML-ul de config și îl întoarce ca dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def slugify(name: str) -> str:
    """Transformă un titlu în slug sigur pentru nume de directoare/fișiere."""
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_').lower()

_LIGATURES = str.maketrans({'ﬁ': 'fi','ﬂ':'fl','’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-'})
_DEHYPHEN = re.compile(r"(\w+)-\n(\w+)")
_LINEBREAK_IN_SENT = re.compile(r"(?<![.!?:\-])\n(?!\n)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

def clean_text(text: str) -> str:
    """Curăță textul OCR: ligaturi, cratime la cap de rând, newline-uri inutile, spații duble."""
    if not text:
        return ""
    text = text.translate(_LIGATURES).replace('\r\n','\n').replace('\r','\n')
    text = _DEHYPHEN.sub(r"\1\2", text)
    text = _LINEBREAK_IN_SENT.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_image_heavy(img: Image.Image, ocr_text: str) -> bool:
    """Heuristic: puțin text raportat la pixeli ⇒ probabil foto/copertă (prag ok pentru 400 DPI)."""
    pixels = img.size[0] * img.size[1]
    return len(ocr_text) < max(400, pixels // 36000)

# ---------- Tesseract path ----------

def _resolve_tesseract_cmd() -> Optional[str]:
    """Setează automat calea către tesseract.exe (TESSERACT_CMD sau locații uzuale pe Windows)."""
    candidates = []
    if os.getenv('TESSERACT_CMD'):
        candidates.append(os.getenv('TESSERACT_CMD'))
    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get('LOCALAPPDATA',''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            print(f"[INFO] Using tesseract: {c}")
            return c
    print("[WARN] Could not find 'tesseract.exe'. Set TESSERACT_CMD env var.")
    return None

# ---------- Preprocesare (PIL) ----------

def crop_margins(pil_img: Image.Image, pct: float = 0.03) -> Image.Image:
    """Taie ~3% din margini (gutter) ca să elimine zgomotul de pe margini."""
    w, h = pil_img.size
    dx, dy = int(w*pct), int(h*pct)
    return pil_img.crop((dx, dy, w-dx, h-dy))

def _pil_preprocess(pil_img: Image.Image, binary: bool, crop_pct: float) -> Image.Image:
    """
    Preproc blând:
    - crop margini (crop_pct), grayscale, UnsharpMask, autocontrast;
    - dacă binary=True (profil 'pil'), aplică o binarizare mediană simplă.
    """
    img = crop_margins(pil_img, crop_pct).convert('L')
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    img = ImageOps.autocontrast(img, cutoff=2)
    if not binary:
        return img
    hist = img.histogram(); total = sum(hist); cum=0; thresh=170
    for i, h in enumerate(hist):
        cum += h
        if cum >= total * 0.5:
            thresh = max(100, min(200, i)); break
    return img.point(lambda p: 255 if p > thresh else 0)

def preprocess_for_ocr(pil_img: Image.Image, mode: str = 'auto', crop_pct: float = 0.03) -> Image.Image:
    """
    Alege profilul de preprocesare:
    - 'pil_gray' -> grayscale fără binarizare (merge excelent pe cărți tipărite)
    - 'pil'      -> ca mai sus + binarizare simplă
    - 'opencv'   -> dacă e instalat, folosește pipeline cv2 + crop (crop_pct)
    - 'auto'     -> preferă 'opencv' dacă există, altfel 'pil_gray'
    """
    m = (mode or 'auto').lower()
    if m == 'pil_gray': return _pil_preprocess(pil_img, binary=False, crop_pct=crop_pct)
    if m == 'pil':      return _pil_preprocess(pil_img, binary=True,  crop_pct=crop_pct)

    # opencv doar dacă există
    try:
        import cv2, numpy as np  # noqa: F401
        HAVE_CV2 = True
    except Exception:
        HAVE_CV2 = False

    if m == 'opencv' and HAVE_CV2:
        from cv2 import fastNlMeansDenoising, adaptiveThreshold, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, createCLAHE, warpAffine, getRotationMatrix2D, INTER_CUBIC, BORDER_REPLICATE, minAreaRect
        import numpy as np, cv2
        arr = np.array(crop_margins(pil_img, crop_pct).convert('L'))
        arr = fastNlMeansDenoising(arr, None, 10, 7, 21)
        arr = createCLAHE(2.0, (8, 8)).apply(arr)
        th = adaptiveThreshold(arr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 15)
        coords = np.column_stack(np.where(th == 0))
        if coords.size > 0:
            angle = minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            h, w = th.shape[:2]
            M = getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            th = warpAffine(th, M, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)
        return Image.fromarray(th)

    # auto fallback
    return _pil_preprocess(pil_img, binary=False, crop_pct=crop_pct)

# ---------- OCR pe pagină ----------

def _avg_conf(pil_img: Image.Image, lang: str, psm: int) -> float:
    """Media de încredere Tesseract pentru un PSM (folosind image_to_data)."""
    cfg = f"--oem 1 --psm {psm}"
    try:
        data = image_to_data(pil_img, lang=lang, config=cfg, output_type=Output.DICT)
        confs = [int(c) for c in data['conf'] if c != '-1']
        return sum(confs)/len(confs) if confs else 0.0
    except Exception:
        return 0.0

def ocr_page(img_path: Path, lang: str, preproc_mode: str, base_psm: int, crop_pct: float) -> tuple[str, float, int]:
    """
    OCR robust:
      - preproc (implicit PIL grayscale cu crop_pct din YAML)
      - rulează image_to_string pentru PSM-urile: [base_psm, 3, 4]
      - alege varianta cu conf. medie mai mare (tie-break: text mai lung)
    Returnează (text_curățat, avg_conf, psm_folosit)
    """
    pil = Image.open(img_path)
    pil = preprocess_for_ocr(pil, preproc_mode, crop_pct=crop_pct)

    tried = []
    for psm in [base_psm, 3, 4]:
        cfg = f"--oem 1 --psm {psm}"
        txt = clean_text(image_to_string(pil, lang=lang, config=cfg))
        conf = _avg_conf(pil, lang, psm)
        tried.append((conf, len(txt), txt, psm))

    tried.sort(key=lambda t: (t[0], t[1]))
    best_conf, _, best_text, best_psm = tried[-1]
    return best_text, best_conf, best_psm

# ---------- Markdown builder ----------

def build_markdown(
    book_title: str,
    pages_dir: Path,
    out_md: Path,
    lang: str,
    embed_images: str = 'auto',
    preproc_mode: str = 'auto',
    tess_psm: int = 6,
    min_chars: int = 120,
    low_conf: float = 55.0,
    crop_pct: float = 0.03,
    from_page: int | None = None,
    to_page:   int | None = None,
):
    """
    Transformă paginile OCR-uite în Markdown:
      - inserează imaginea (auto/all/none)
      - elimină textul junk pe paginile slabe (len<min_chars sau conf<low_conf)
      - suportă interval de pagini (1-based, inclusiv)
    """
    pages = sorted(glob.glob(str(pages_dir / 'page_*.jpg')))
    if from_page or to_page:
        start = 1 if from_page is None else max(1, from_page)
        end   = len(pages) if to_page is None else min(len(pages), to_page)
        pages = pages[start-1:end]
        start_number = start
    else:
        start_number = 1

    out_md.parent.mkdir(parents=True, exist_ok=True)
    qa_path = out_md.parent / "qa_report.csv"
    with open(out_md, 'w', encoding='utf-8') as md, \
         open(qa_path, 'w', newline='', encoding='utf-8') as qaf:
        writer = csv.writer(qaf)
        writer.writerow(['page','conf','chars','psm','embedded','preprocess','crop_pct'])

        md.write(f"# {book_title}\n\n")
        md.write(f"> OCR language: `{lang}` | Preprocess: `{preproc_mode}` | Base PSM: `{tess_psm}` | Crop: `{crop_pct}`\n\n")
        md.write("---\n\n")

        for idx, p in enumerate(tqdm(pages, desc='[OCR] Pages → Markdown', unit='page'), start=start_number):
            p_path = Path(p)
            img = Image.open(p_path)
            text, conf, used_psm = ocr_page(p_path, lang, preproc_mode, tess_psm, crop_pct)

            # decizie inserare imagine
            embed_decision = (embed_images == 'all') or (embed_images == 'auto' and ((idx == 1) or is_image_heavy(img, text)))

            # embed-only dacă OCR e slab
            low_quality = (len(text) < min_chars) or (conf < low_conf)
            did_embed = embed_decision or (embed_images != 'none' and low_quality)

            # CSV: log QA
            writer.writerow([idx, round(conf, 1), len(text), used_psm, int(did_embed), preproc_mode, crop_pct])

            if low_quality:
                if embed_images != 'none':
                    rel = os.path.relpath(p_path, out_md.parent).replace(os.sep, '/')
                    md.write(f"![Page {idx}]({rel})\n\n")
                md.write(f"## Page {idx}\n\n*(Low-confidence OCR; image embedded only.)*\n\n---\n\n")
                continue

            if did_embed and embed_decision:
                rel = os.path.relpath(p_path, out_md.parent).replace(os.sep, '/')
                md.write(f"![Page {idx}]({rel})\n\n")

            md.write(f"## Page {idx}\n\n{txt_or(text)}\n\n---\n\n")

def txt_or(s: str) -> str:
    """Mic helper ca să evităm linii goale nejustificate."""
    return s if s.strip() else "*(No detectable text on this page.)*"

# ---------- Entry point ----------

def main():
    """Parsează CLI, citește YAML-ul, pregătește căile și rulează conversia în Markdown."""
    _resolve_tesseract_cmd()

    import argparse
    parser = argparse.ArgumentParser(description='OCR pages → Markdown')
    parser.add_argument('-c','--config', default='configs/families_crisis.yaml', help='YAML config file')
    parser.add_argument('--preprocess', choices=['auto','pil','pil_gray','opencv'], default=None,
                        help='Override preprocess mode (default: from YAML or auto)')
    parser.add_argument('--from-page', type=int, default=None, help='Start page (1-based)')
    parser.add_argument('--to-page',   type=int, default=None, help='End page (1-based, inclusive)')
    parser.add_argument('--tess-psm',  type=int, default=None, help='Override tess_psm')
    parser.add_argument('--crop-pct',  type=float, default=None, help='Override crop_pct (e.g. 0.05)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    title       = cfg['book_title']
    lang        = cfg.get('lang','eng')
    embed_imgs  = cfg.get('embed_images','auto')
    pre_mode    = args.preprocess or cfg.get('preprocess','pil_gray')  # implicit pil_gray
    tess_psm    = int(args.tess_psm if args.tess_psm is not None else cfg.get('tess_psm', 6))
    crop_pct    = float(args.crop_pct if args.crop_pct is not None else cfg.get('crop_pct', 0.03))
    min_chars   = int(cfg.get('min_chars', 120))
    low_conf    = float(cfg.get('low_conf', 55.0))

    book_slug = slugify(title)
    base_out  = Path('output') / book_slug
    pages_dir = base_out / 'split_pages'
    out_md    = base_out / 'markdown' / f"{book_slug}.md"

    if not pages_dir.exists():
        raise SystemExit(f"Split pages not found: {pages_dir}. Run process_pdf.py first.")

    print(f"[INFO] Building Markdown for '{title}' ({lang}) from {pages_dir} …")
    build_markdown(
        title, pages_dir, out_md, lang,
        embed_images=embed_imgs,
        preproc_mode=pre_mode,
        tess_psm=tess_psm,
        min_chars=min_chars,
        low_conf=low_conf,
        crop_pct=crop_pct,
        from_page=args.from_page,
        to_page=args.to_page,
    )
    print(f"[DONE] Markdown saved → {out_md}")

if __name__ == '__main__':
    main()
