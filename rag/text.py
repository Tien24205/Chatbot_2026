"""
Hằng số và regex dùng chung cho việc đọc văn bản tiếng Việt.

Gom về đây những mẫu từng được định nghĩa lặp ở nhiều file. Cùng một regex nằm
ba chỗ nghĩa là ba chỗ phải nhớ sửa khi knowledge base đổi cách viết — và chỉ
cần quên một chỗ là tầng chunking với tầng đồ thị hiểu tài liệu khác nhau.
"""

from __future__ import annotations

import re

# Chữ HOA tiếng Việt. Không dùng được lớp ký tự kiểu [A-ZĐÀ-Ỹ]: khoảng À-Ỹ trong
# Unicode chứa xen kẽ cả chữ thường (ỗ, ộ, ề... đều nằm giữa À và Ỹ), nên lớp đó
# khớp luôn chữ thường — đo được: "mỗi đội" bị nhận nhầm là danh từ riêng.
# Khối U+1EA0..U+1EF8 xếp xen kẽ hoa/thường, mã CHẴN là chữ hoa.
UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯ" + "".join(chr(c) for c in range(0x1EA0, 0x1EFA, 2))

# Tiêu đề markdown: "## Mục tiêu trung lập" -> ("##", "Mục tiêu trung lập").
# Dùng ở chunker (cắt theo mục) và graph (biết thực thể thuộc mục nào).
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Đầu một mục danh sách: "- ", "* ", "• ", "3. ".
# Knowledge base viết định nghĩa theo đúng khuôn này, nên nó vừa là ranh giới
# không được xé đôi khi chunk, vừa là dấu hiệu để trích thực thể.
BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
