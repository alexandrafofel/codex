project_root/
│
├── Folder_Structure.md
│
├── input/
│   └── families_crisis.pdf          # Fișierul original
│
├── output/
│   └── families_crisis/
│       ├── raw_pages/               # imagini extrase din PDF
│       │   ├── scan_001.jpg
│       │   ├── scan_002.jpg
│       │   └── ...
│       └── split_pages/             # imagini separate (1 pagină reală per fișier)
│           ├── page_001.jpg
│           ├── page_002.jpg
│           └── ...
│
├── configs/
│   └── families_crisis.yaml         # config specific per carte
│
└── process_pdf.py                   # script principal
