# Detailed Design

# 1) Scope & obiective

- Produce **două fișiere .md**:
    1. **Text-only**: fără imagini, după preprocesare „albire” pentru pagini predominant text.
    2. **Images-only**: inserează imagini croite fidel în pozițiile de pagină, cu nume determinist.
- Flux „în doi pași” cu **feedback Y/N** după generarea textului.
- Evită „junk text”: gating pe `avg_conf` și `chars`, plus post-procesare.
- Nu reteza titluri/footere: **fără crop procentual orb** în mod implicit; ulterior, „safe-crop” conștient de text.

# 2) Config & convenții

## Fișiere YAML

- `configs/families_crisis_text.yaml` (profil text-only)
    - `preprocess: opencv`
    - `embed_images: none`
    - `tess_psm: 4`
    - `crop_pct: 0.00`
    - `low_conf: 65`, `min_chars: 180`
- `configs/families_crisis_images.yaml` (profil images-only)
    - `preprocess: pil_gray`
    - `embed_images: all`
    - `tess_psm: 6` (ignorăm textul prin gating extrem: `low_conf=999`, `min_chars=10000`)
    - `crop_pct: 0.00`

## Layout output

```
output/<slug>/
  raw_pages/
  split_pages/
  markdown/
    img/
    <slug>_TEXT.md
    <slug>_IMAGES.md

```

## Denumire imagini

- Format: `page_{NNN}_pxpagini.jpg`
    - `{NNN}` = index pagină 1-based, zero-padded la 3 (`001`, `002`, …).
    - `pxpagini`:
        - `1x1` pentru pagini single;
        - `1x2`, `2x2` pentru split-detect (dublu, quadruplu).
    - Exemplu: `page_001_1x1.jpg`, `page_015_1x2.jpg`.

# 3) CLI & UX

## Subcomenzi existente

- `pdf`, `split`, `ocr`, `md`, `all` (rămân)

## Comportament nou

- `all -c <text-config>`:
    - rulează `pdf → split → ocr (după config text) → md (text-only)`.
    - La final: prompt **`Generate images MD using <images-config>? [Y/N]`**
        - `Y`: rulează `md` **în modul imagini** (fără re-OCR), folosind `configs/families_crisis_images.yaml`.
            
            (Config de imagini poate fi dat prin `--images-config`; default naming: înlocuiește `_text` cu `_images` dacă există.)
            
        - `N`: se oprește.

## Parametri noi (opțional, utile)

- `md --images -c <images-config>` → generează direct **IMAGES.md** + `img/`.
- `all --images-config <path>` → suprascrie valoarea implicită.

# 4) Module & interfețe

## `config.py`

- `load_config(path: str) -> OcrConfig`
- `OcrConfig` (câmpuri cheie):
    - `book_title: str`, `pdf_file: str`, `lang: str`, `dpi: int`
    - `preprocess: Literal["pil_gray","pil_bin","opencv"]`
    - `tess_psm: int`
    - `embed_images: Literal["auto","all","none"]`
    - `crop_pct: float`
    - `low_conf: int`
    - `min_chars: int`
    - `split_strategy: list[SplitRule]`

## `pdf.py`

- `extract_pages(cfg: OcrConfig, work_dir: Path) -> list[Path]`
    - PyMuPDF la DPI ales; fallback pdf2image + Poppler.
    - Output: `raw_pages/scan_XXX.jpg|png`.

## `split.py`

- `split_pages(cfg: OcrConfig, work_dir: Path, raw_pages: list[Path]) -> list[Path]`
    - `detect_gutter(img) -> SplitType`
    - `auto_orient_deskew(img) -> Image`
    - Scrie în `split_pages/page_{NNN}.jpg`.

## `preprocess.py`

- `preprocess_image(img: Image, cfg: OcrConfig) -> Image`
    - `pil_gray`: grayscale + unsharp + autocontrast.
    - `pil_bin`: Otsu.
    - `opencv`: CLAHE + adaptive threshold + deskew (mai agresiv).
- **(later)** `safe_crop(img) -> Image`:
    - construiește content mask, evaluează benzi; nu taie dacă detectează text; bbox + padding.

## `ocr.py`

- `run_ocr(img: Image, cfg: OcrConfig) -> OcrResult`
    - rulează multi-PSM `[cfg.tess_psm, 6, 3, 4]`.
    - `OcrResult(text: str, avg_conf: float, used_psm: int, chars: int)`.
- `is_image_heavy(img, dpi, text_len) -> bool`
    - euristică pe pixel count & lungime text.

## `structure.py`

- `detect_headings(dataframe_like) -> list[Block]`
    - linii cu înălțime în quantil ≥0.85; centrare & poziție în treimea superioară → H1.
- `segment_blocks(text, heading_meta) -> list[Block]`
    - `Block` tip: `heading/body/figure/caption`.

## `markdown.py` (extins)

- `build_markdown(cfg: OcrConfig, page_paths: list[Path], work_dir: Path) -> Path`
    - decide **ce variantă** să cheme în funcție de `cfg.embed_images`.
- `render_text_only(pages: list[PageCtx], out_dir: Path, slug: str) -> Path`
    - scrie `<slug>_TEXT.md`; **nu** salvează imagini.
- `render_with_images(pages: list[PageCtx], out_dir: Path, slug: str) -> Path`
    - salvează fiecare imagine în `markdown/img/page_{NNN}_pxpagini.jpg`.
    - scrie `<slug>_IMAGES.md` cu link-uri `![caption](img/page_...jpg)`.
- `cleanup_text(s: str) -> str`
    - de-hyphen (la line wrap), normalize whitespace, elimină „|” rătăcite.
- **Gating** (aplicat înainte de rendere):
    - dacă `avg_conf < low_conf` sau `chars < min_chars` ⇒ în **Text-only**: pagina devine goală; în **Images-only**: pagina doar imagine.

### `PageCtx` (structură folosită în markdown builder)

- `index: int` (1-based)
- `split_tag: str` (`1x1`, `1x2`, `2x2`)
- `preprocessed_img: Image` (optional; doar când trebuie scrise imagini)
- `ocr: OcrResult` (optional în images-only)
- `blocks: list[Block]` (din `structure.py`)
- `embed: bool` (decis de `cfg.embed_images` și gating)

## `cli.py` (punctele-cheie)

- `cmd_all(args)`:
    - `cfg_text = load_config(args.config)`
    - `pages = extract → split → ocr (după cfg_text)`
    - `build_markdown(cfg_text, pages, work_dir)` → `<slug>_TEXT.md`
    - Prompt: `Generate images MD using <images-config>? [Y/N]`
        - dacă `Y`: `cfg_img = load_config(images_cfg)`; `build_markdown(cfg_img, pages, work_dir)` (fără re-OCR) → `<slug>_IMAGES.md`
- `cmd_md(args)`:
    - dacă `-images` sau `cfg.embed_images == "all"` → `render_with_images`, altfel `render_text_only`.

# 5) Algoritmi & decizii

## Gating „no junk”

```python
def allow_text(ocr: OcrResult, cfg: OcrConfig) -> bool:
    if ocr.avg_conf < cfg.low_conf:
        return False
    if ocr.chars < cfg.min_chars:
        return False
    return True

```

- **Text-only**: `allow_text == True` → text în MD; altfel pagină goală (fără imagine).
- **Images-only**: ignorăm textul; inserăm imaginea (nu vrem „albire” agresivă pe fotografii).

## Nume imagine & link MD

```python
img_name = f"page_{index:03d}_{split_tag}.jpg"  # ex: page_002_1x1.jpg
img_path = out_img_dir / img_name
md_line = f"![page {index}](img/{img_name})"

```

## „Safe-crop” (future work; opțional acum)

- Calculează text mask (OpenCV: adaptive threshold).
- Dilate + remove noise; ia bbox maxim al conținutului.
- Păstrează padding (ex. 16 px); **nu** cropezi dacă în benzile candidate există text (verificat prin `ocr-lite` PSM 7 pe benzi).

# 6) Erori & mesaje

- Lipsă PDF → mesaj clar cu calea absolută cerută.
- Lipsă Poppler când fallback → instrucțiuni instalare.
- Tesseract absent → `TESSERACT_CMD` și `TESSDATA_PREFIX` guidance.
- La `md`:
    - `Text-only`: la final, log cu `% pagini cu text`, `% pagini goale`.
    - Prompt Y/N (timeout opțional → N).

# 7) Performanță & cache

- Cache minimal pe pagină: `.json` cu `hash(raw_img)` + `psm_used` + `avg_conf` → sări peste OCR dacă identic.
- Paralelizare opțională (batch pe PSM) → la Windows rămânem pe secvențial pentru stabilitate (Tesseract concurent poate fi instabil).
- `dpi`: 300 pentru iterare rapidă; 400 pentru „final”.

# 8) Testare

- Unit:
    - `config.load_config` (validări).
    - Denumire imagine (`page_`, `split_tag`).
    - `cleanup_text` (de-hyphen, whitespace).
    - Gating logic (`allow_text`).
- Integrare:
    - `ocr` pe 2–3 pagini sintetice.
    - `render_text_only` și `render_with_images` pe fixture-uri.
- QA scripts:
    - `scripts/eval_cer.py`: CER/WER + F1(headings) pe gold de 50 pagini.

# 9) CI (GitHub Actions)

- Instalează `tesseract-ocr`, `poppler-utils`.
- `pip install -e .[dev]`.
- Rulează `ruff`, `mypy`, `pytest`.
- Dacă detectează `configs/*_text.yaml` + `input/*.pdf`:
    - rulează `ocrmd all -c <first_text_cfg>` **fără** prompt (setează `-non-interactive` să sară feedback).
    - urcă artefactele: `<slug>_TEXT.md` (+ opțional `<slug>_IMAGES.md` dacă e `-images-config` setat).

# 10) Plan de migrare (incremental)

1. Extinde `markdown.py` cu `render_text_only`, `render_with_images`, nume imagini, două fișiere MD.
2. Extinde `cli.py`:
    - `-images-config`
    - prompt Y/N (cu `-non-interactive` pentru CI).
3. Ajustează `build_markdown` să **NU** forțeze re-OCR dacă există rezultate (primește `pages`/`PageCtx` cu OCR deja rulat).
4. Adaugă gating nou în `markdown.py` înainte de randare.
5. (Opțional) implementează `safe_crop` și activează-l pentru `preprocess: opencv`.