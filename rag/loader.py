"""
BƯỚC 2 — Document loader & cleaning.

Đọc file trong knowledge_base/ và chuẩn hoá text trước khi chunk.
Không phụ thuộc API key — chạy được offline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """Một tài liệu đã được làm sạch."""

    source_name: str  # tên file, dùng làm nguồn trích dẫn
    path: Path
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def clean_text(raw: str) -> str:
    """
    Làm sạch text thô.

    Xử lý các vấn đề plan §Bước 2 liệt kê:
      - encoding: chuẩn hoá Unicode về NFC (tiếng Việt có 2 cách gõ dấu khác nhau,
        nếu không chuẩn hoá thì "hoà" và "hoà" là 2 chuỗi khác nhau với máy);
      - ký tự đặc biệt: bỏ BOM, zero-width space, non-breaking space;
      - xuống dòng sai: thống nhất CRLF/CR về LF;
      - khoảng trắng thừa: bỏ trailing space, gom >2 dòng trống thành 1 dòng trống.
    """
    text = raw.replace("﻿", "")  # BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # NFC: bắt buộc với tiếng Việt, nếu không embedding sẽ coi 2 cách gõ là 2 từ.
    text = unicodedata.normalize("NFC", text)

    # Khoảng trắng "ẩn" hay lọt vào khi copy từ web/PDF.
    text = text.replace("​", "").replace(" ", " ")

    # Bỏ khoảng trắng cuối dòng.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Gom nhiều dòng trống liên tiếp thành đúng một dòng trống (ranh giới đoạn).
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_file(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    return Document(source_name=path.name, path=path, text=clean_text(raw))


def load_directory(directory: Path) -> list[Document]:
    """Đọc toàn bộ file được hỗ trợ trong thư mục, sắp xếp theo tên."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục: {directory}")

    paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            f"Thư mục {directory} không có file .txt hoặc .md nào."
        )

    return [load_file(p) for p in paths]
