"""
图像批量重命名工具 (高速版)
Author: Auorui
Blog: https://blog.csdn.net/m0_62919535/article/details/166844021
"""
from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading

from PIL import Image, ImageOps
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

FORMAT_OPTIONS = {
    "保持原格式": None,
    "JPG (*.jpg)": ("JPEG", ".jpg"),
    "JPEG (*.jpeg)": ("JPEG", ".jpeg"),
    "PNG (*.png)": ("PNG", ".png"),
    "WEBP (*.webp)": ("WEBP", ".webp"),
    "BMP (*.bmp)": ("BMP", ".bmp"),
    "TIFF (*.tiff)": ("TIFF", ".tiff"),
    "TIF (*.tif)": ("TIFF", ".tif"),
}


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    mode: str
    image_format: str
    file_size: int


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def mode_shape(mode: str, width: int, height: int) -> str:
    """获取图像形状信息，兼容Pillow 10+"""
    try:
        if hasattr(Image, 'getmodebands'):
            bands = Image.getmodebands(mode)
            if isinstance(bands, int):
                channels = bands
            else:
                channels = len(bands)
        else:
            channels = len(mode)
    except Exception:
        channels = len(mode)

    if channels == 1:
        return f"({height}, {width})"
    else:
        return f"({height}, {width}, {channels})"


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


class DropTable(QTableWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        self.files_dropped.emit(paths)
        event.acceptProposedAction()


class ImageProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, records, names, output_dir, format_combo, overwrite_checkbox):
        super().__init__()
        self.records = records
        self.names = names
        self.output_dir = output_dir
        self.format_combo = format_combo
        self.overwrite_checkbox = overwrite_checkbox
        self.is_running = True
        self._lock = threading.Lock()
        self._processed = 0
        self._errors = []
        self._format_option = FORMAT_OPTIONS[format_combo]

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            total = len(self.records)
            self._processed = 0
            self._errors = []

            if total == 0:
                self.finished.emit()
                return

            # 动态调整线程数：图片少时用少线程避免开销，图片多时用多线程
            if total < 10:
                max_workers = max(1, min(os.cpu_count() or 2, total))
            else:
                max_workers = min(os.cpu_count() * 2 or 4, 16, total)

            # 如果只是复制文件（保持原格式），可以使用更多线程
            if self._format_option is None:
                max_workers = min(max_workers * 2, 32)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for row, (record, name) in enumerate(zip(self.records, self.names)):
                    if not self.is_running:
                        break
                    future = executor.submit(
                        self.process_single_image,
                        row, record, name
                    )
                    futures[future] = row

                for future in as_completed(futures):
                    if not self.is_running:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        row, error = future.result()
                        with self._lock:
                            self._processed += 1
                            progress = int(self._processed / total * 100)
                            self.progress.emit(progress)
                            self.status.emit(f"处理中 {self._processed}/{total}")
                            if error:
                                self._errors.append(error)
                    except Exception as e:
                        with self._lock:
                            self._errors.append(str(e))

            if self._errors:
                error_count = len(self._errors)
                for err in self._errors[:5]:
                    self.error.emit(err)
                if error_count > 5:
                    self.error.emit(f"... 还有 {error_count - 5} 个错误")

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"程序错误: {str(e)}")
            self.finished.emit()

    def process_single_image(self, row: int, record: ImageRecord, name: str) -> tuple[int, str | None]:
        """处理单张图片，返回 (row, error_message)"""
        destination = self.output_dir / name

        # 检查文件是否已存在
        if destination.exists() and not self.overwrite_checkbox:
            return row, f"跳过 {name} (文件已存在)"

        try:
            self.write_image(record, destination)
            return row, None
        except Exception as exc:
            return row, f"{record.path.name}: {exc}"

    def write_image(self, record: ImageRecord, destination: Path) -> None:
        format_option = self._format_option

        # 如果是保持原格式，直接复制文件（最快）
        if format_option is None:
            if record.path == destination:
                raise ValueError("输入文件和输出文件不能相同")
            shutil.copy2(record.path, destination)
            return

        target_format = format_option[0]

        # 优化：如果目标格式和原格式相同，直接复制
        if target_format == record.image_format.upper():
            if record.path == destination:
                raise ValueError("输入文件和输出文件不能相同")
            shutil.copy2(record.path, destination)
            return

        with Image.open(record.path) as source:
            source.seek(0)
            image = ImageOps.exif_transpose(source)
            save_kwargs: dict[str, object] = {}

            # 保持原有质量参数不变
            if target_format == "JPEG":
                image = self.flatten_for_opaque_format(image, background=(255, 255, 255))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                save_kwargs.update(quality=95, optimize=True)
            elif target_format == "WEBP":
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                save_kwargs.update(quality=95, method=6)
            elif target_format == "PNG":
                if image.mode == "CMYK":
                    image = image.convert("RGB")
                save_kwargs.update(optimize=True)
            elif target_format == "BMP":
                image = self.flatten_for_opaque_format(image, background=(255, 255, 255))
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
            elif target_format == "TIFF" and image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
                save_kwargs.update(compression="tiff_deflate")

            image.save(destination, format=target_format, **save_kwargs)

    @staticmethod
    def flatten_for_opaque_format(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
        has_alpha = image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
        )
        if not has_alpha:
            return image
        rgba = image.convert("RGBA")
        base = Image.new("RGB", rgba.size, background)
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base


class ImageBatchRenamer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.path_keys: set[str] = set()
        self.processor: ImageProcessor | None = None
        self.last_open_dir: str = str(Path.home())
        self.last_output_dir: str = str(Path.home() / "Desktop" / "renamed_images")
        self._image_cache: dict[str, QPixmap] = {}  # 预览缓存

        self.setWindowTitle("图像批量重命名工具 (高速版)")
        self.setMinimumSize(1080, 700)
        self.resize(1320, 820)

        self._build_ui()
        self._apply_style()
        self._connect_signals()
        self.update_name_preview()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(18, 12, 18, 12)
        toolbar_layout.setSpacing(8)

        self.add_files_button = QPushButton("选择图像")
        self.add_folder_button = QPushButton("选择文件夹")
        self.remove_button = QPushButton("删除选中")
        self.clear_button = QPushButton("清空列表")
        for button in (
                self.add_files_button,
                self.add_folder_button,
                self.remove_button,
                self.clear_button,
        ):
            button.setMinimumHeight(36)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        toolbar_layout.addWidget(self.add_files_button)
        toolbar_layout.addWidget(self.add_folder_button)
        toolbar_layout.addWidget(self.remove_button)
        toolbar_layout.addWidget(self.clear_button)
        toolbar_layout.addStretch()
        self.count_label = QLabel("共 0 张图像")
        self.count_label.setObjectName("mutedLabel")
        toolbar_layout.addWidget(self.count_label)
        root.addWidget(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 10, 16)
        left_layout.setSpacing(10)

        hint = QLabel("可将图像或文件夹直接拖入下方列表")
        hint.setObjectName("mutedLabel")
        left_layout.addWidget(hint)

        self.table = DropTable()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["原文件名", "尺寸", "形状", "模式", "格式", "大小", "新文件名"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 80)
        self.table.setColumnWidth(5, 100)

        self.table.setMinimumWidth(700)
        left_layout.addWidget(self.table, 1)

        self.selection_info = QLabel("尚未选择图像")
        self.selection_info.setObjectName("mutedLabel")
        left_layout.addWidget(self.selection_info)
        splitter.addWidget(left)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setMinimumWidth(360)
        settings_scroll.setMaximumWidth(430)
        settings_scroll.setObjectName("settingsScroll")

        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(18, 16, 18, 20)
        settings_layout.setSpacing(14)

        title = QLabel("重命名设置")
        title.setObjectName("panelTitle")
        settings_layout.addWidget(title)

        naming_group = QGroupBox("文件名规则")
        naming_form = QFormLayout(naming_group)
        naming_form.setContentsMargins(14, 18, 14, 14)
        naming_form.setHorizontalSpacing(12)
        naming_form.setVerticalSpacing(12)

        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("可选前缀")

        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("可选后缀")

        self.number_length_spin = QSpinBox()
        self.number_length_spin.setRange(0, 12)
        self.number_length_spin.setValue(0)
        self.number_length_spin.setToolTip("设为0则不添加数字序号")

        self.start_number_spin = QSpinBox()
        self.start_number_spin.setRange(0, 999_999_999)
        self.start_number_spin.setValue(0)
        self.start_number_spin.setToolTip("起始编号（数字长度为0时无效）")

        naming_form.addRow("前缀", self.prefix_edit)
        naming_form.addRow("后缀", self.suffix_edit)
        naming_form.addRow("数字长度", self.number_length_spin)
        naming_form.addRow("起始数", self.start_number_spin)
        settings_layout.addWidget(naming_group)

        format_group = QGroupBox("输出格式")
        format_layout = QVBoxLayout(format_group)
        format_layout.setContentsMargins(14, 18, 14, 14)
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_OPTIONS.keys())
        self.format_combo.setCurrentText("保持原格式")
        format_layout.addWidget(self.format_combo)
        self.format_note = QLabel("保持每张图像当前的扩展名，不重新编码。")
        self.format_note.setWordWrap(True)
        self.format_note.setObjectName("mutedLabel")
        format_layout.addWidget(self.format_note)
        settings_layout.addWidget(format_group)

        output_group = QGroupBox("输出位置")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(14, 18, 14, 14)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(self.last_output_dir)
        self.output_button = QPushButton("浏览")
        self.output_button.setFixedWidth(62)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        output_layout.addLayout(output_row)

        self.timestamp_checkbox = QCheckBox("在输出目录下创建时间戳子文件夹")
        self.timestamp_checkbox.setChecked(False)
        self.timestamp_checkbox.setToolTip("例如: 2026_08_10_14_30_25")
        output_layout.addWidget(self.timestamp_checkbox)

        self.recursive_checkbox = QCheckBox("选择文件夹时包含子文件夹")
        self.recursive_checkbox.setChecked(True)
        self.overwrite_checkbox = QCheckBox("允许覆盖输出目录中的同名文件")
        self.overwrite_checkbox.setChecked(False)
        output_layout.addWidget(self.recursive_checkbox)
        output_layout.addWidget(self.overwrite_checkbox)
        settings_layout.addWidget(output_group)

        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(14, 18, 14, 14)
        self.preview_label = QLabel("选择列表中的图像")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(165)
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preview_meta = QLabel("")
        self.preview_meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_meta.setWordWrap(True)
        self.preview_meta.setObjectName("mutedLabel")
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.preview_meta)
        settings_layout.addWidget(preview_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(25)
        settings_layout.addWidget(self.progress_bar)

        self.start_button = QPushButton("开始批量处理")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumHeight(48)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_layout.addWidget(self.start_button)

        settings_layout.addStretch()

        settings_scroll.setWidget(settings)
        splitter.addWidget(settings_scroll)
        splitter.setSizes([940, 380])

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

        self.number_length_spin.valueChanged.connect(self.toggle_start_number)

    def toggle_start_number(self, value: int) -> None:
        self.start_number_spin.setEnabled(value > 0)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f5f7fa; color: #20242a; }
            QWidget { font-family: "Microsoft YaHei UI"; font-size: 13px; }
            QFrame#toolbar { background: #ffffff; border-bottom: 1px solid #dfe3e8; }
            QLabel#panelTitle { font-size: 19px; font-weight: 600; color: #16191d; padding: 2px 0 3px 0; }
            QLabel#mutedLabel { color: #6c7580; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dfe3e8;
                border-radius: 7px;
                margin-top: 9px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                color: #343a40;
                background: #f5f7fa;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cfd5dc;
                border-radius: 5px;
                padding: 7px 13px;
            }
            QPushButton:hover { border-color: #2f78ff; color: #145ddd; background: #f5f9ff; }
            QPushButton:pressed { background: #eaf2ff; }
            QPushButton#primaryButton {
                background: #246bfd;
                border-color: #246bfd;
                color: white;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover { background: #155ae6; border-color: #155ae6; color: white; }
            QPushButton#primaryButton:disabled {
                background: #a0b8e8;
                border-color: #a0b8e8;
            }
            QLineEdit, QSpinBox, QComboBox {
                background: #ffffff;
                border: 1px solid #cfd5dc;
                border-radius: 5px;
                min-height: 32px;
                padding: 0 8px;
                selection-background-color: #246bfd;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #246bfd; }
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #dfe3e8;
                border-radius: 6px;
                selection-background-color: #e5efff;
                selection-color: #20242a;
            }
            QHeaderView::section {
                background: #eef1f5;
                color: #4c5661;
                border: none;
                border-bottom: 1px solid #d8dde3;
                padding: 8px 6px;
                font-weight: 600;
            }
            QLabel#previewLabel {
                background: #edf1f5;
                border: 1px solid #d9dee5;
                border-radius: 5px;
                color: #76808b;
            }
            QScrollArea#settingsScroll { border: none; border-left: 1px solid #dfe3e8; }
            QCheckBox { spacing: 8px; }
            QStatusBar { background: #ffffff; border-top: 1px solid #dfe3e8; color: #68717c; }
            QProgressBar {
                border: 1px solid #d0d7de;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #246bfd;
                border-radius: 4px;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.add_files_button.clicked.connect(self.choose_files)
        self.add_folder_button.clicked.connect(self.choose_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_records)
        self.output_button.clicked.connect(self.choose_output_directory)
        self.start_button.clicked.connect(self.process_images)
        self.table.files_dropped.connect(self.add_dropped_paths)
        self.table.itemSelectionChanged.connect(self.show_selected_preview)

        self.prefix_edit.textChanged.connect(self.update_name_preview)
        self.suffix_edit.textChanged.connect(self.update_name_preview)
        self.number_length_spin.valueChanged.connect(self.update_name_preview)
        self.start_number_spin.valueChanged.connect(self.update_name_preview)
        self.format_combo.currentTextChanged.connect(self.on_format_changed)

    def choose_files(self) -> None:
        filters = "图像文件 (*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff);;所有文件 (*.*)"
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像",
            self.last_open_dir,
            filters
        )
        if names:
            self.last_open_dir = str(Path(names[0]).parent)
            self.add_paths([Path(name) for name in names])

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择图像文件夹",
            self.last_open_dir
        )
        if not folder:
            return
        self.last_open_dir = folder
        root = Path(folder)
        iterator = root.rglob("*") if self.recursive_checkbox.isChecked() else root.glob("*")
        self.add_paths([path for path in iterator if path.is_file()])

    def add_dropped_paths(self, paths: list[Path]) -> None:
        expanded: list[Path] = []
        for path in paths:
            if path.is_dir():
                iterator = path.rglob("*") if self.recursive_checkbox.isChecked() else path.glob("*")
                expanded.extend(item for item in iterator if item.is_file())
            else:
                expanded.append(path)
        if expanded:
            self.last_open_dir = str(expanded[0].parent)
        self.add_paths(expanded)

    def add_paths(self, paths: list[Path]) -> None:
        candidates = [path for path in paths if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        added = 0
        failed: list[str] = []

        total = len(candidates)
        self.statusBar().showMessage(f"正在扫描图像... 0/{total}")

        for i, path in enumerate(candidates):
            # 每10张更新一次进度，避免UI卡顿
            if i % 10 == 0:
                self.statusBar().showMessage(f"正在扫描图像... {i}/{total}")
                QApplication.processEvents()

            try:
                resolved = path.resolve()
                key = os.path.normcase(str(resolved))
                if key in self.path_keys:
                    continue
                with Image.open(resolved) as image:
                    width, height = image.size
                    record = ImageRecord(
                        path=resolved,
                        width=width,
                        height=height,
                        mode=image.mode,
                        image_format=(image.format or resolved.suffix.lstrip(".")).upper(),
                        file_size=resolved.stat().st_size,
                    )
                self.records.append(record)
                self.path_keys.add(key)
                added += 1
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

        self.refresh_table()
        if failed:
            details = "\n".join(failed[:8])
            if len(failed) > 8:
                details += f"\n……另有 {len(failed) - 8} 个文件"
            QMessageBox.warning(self, "部分文件无法读取", details)
        self.statusBar().showMessage(f"已添加 {added} 张图像", 4000)

    def refresh_table(self) -> None:
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            values = [
                record.path.name,
                f"{record.width}×{record.height}",
                mode_shape(record.mode, record.width, record.height),
                record.mode,
                record.image_format,
                human_size(record.file_size),
                "",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(record.path) if column == 0 else value)
                if column in (1, 2, 3, 4, 5):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        self.count_label.setText(f"共 {len(self.records)} 张图像")
        self.update_name_preview()

    def desired_extension(self, record: ImageRecord) -> str:
        option = FORMAT_OPTIONS[self.format_combo.currentText()]
        if option is None:
            return record.path.suffix.lower()
        return option[1]

    def output_name(self, row: int, record: ImageRecord) -> str:
        prefix = self.prefix_edit.text().strip()
        suffix = self.suffix_edit.text().strip()
        digit_length = self.number_length_spin.value()
        original_stem = record.path.stem

        if digit_length > 0:
            number = self.start_number_spin.value() + row
            padded = str(number).zfill(digit_length)

            if prefix and suffix:
                base_name = f"{prefix}{padded}{suffix}"
            elif prefix:
                base_name = f"{prefix}{padded}"
            elif suffix:
                base_name = f"{padded}{suffix}"
            else:
                base_name = padded
        else:
            if prefix and suffix:
                base_name = f"{prefix}{original_stem}{suffix}"
            elif prefix:
                base_name = f"{prefix}{original_stem}"
            elif suffix:
                base_name = f"{original_stem}{suffix}"
            else:
                return f"{original_stem}{self.desired_extension(record)}"

        base_name = self.clean_name_part(base_name)

        if not base_name:
            base_name = original_stem

        return f"{base_name}{self.desired_extension(record)}"

    @staticmethod
    def clean_name_part(value: str) -> str:
        invalid = '<>:"/\\|?*'
        result = value.strip()
        for char in invalid:
            result = result.replace(char, "_")
        return result

    def update_name_preview(self) -> None:
        seen: set[str] = set()
        for row, record in enumerate(self.records):
            name = self.output_name(row, record)
            item = self.table.item(row, 6)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, 6, item)
            item.setText(name)
            key = name.casefold()
            duplicate = key in seen
            seen.add(key)
            item.setForeground(QColor("#c62828") if duplicate else QColor("#1559c4"))
            item.setToolTip("名称重复，请调整编号设置" if duplicate else name)

    def on_format_changed(self, text: str) -> None:
        if FORMAT_OPTIONS[text] is None:
            self.format_note.setText("保持每张图像当前的扩展名，不重新编码。")
        else:
            self.format_note.setText("图像将重新编码为所选格式；动画 GIF 只处理第一帧。")
        self.update_name_preview()

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            record = self.records.pop(row)
            self.path_keys.discard(os.path.normcase(str(record.path)))
        self.refresh_table()
        self.clear_preview_if_empty()

    def clear_records(self) -> None:
        if self.records:
            answer = QMessageBox.question(self, "清空列表", "确定要移除列表中的所有图像吗？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.records.clear()
        self.path_keys.clear()
        self._image_cache.clear()  # 清空缓存
        self.refresh_table()
        self.clear_preview_if_empty()

    def clear_preview_if_empty(self) -> None:
        if not self.records or not self.table.selectedItems():
            self.preview_label.clear()
            self.preview_label.setText("选择列表中的图像")
            self.preview_meta.clear()
            self.selection_info.setText("尚未选择图像")

    def choose_output_directory(self) -> None:
        current = self.output_edit.text().strip() or self.last_output_dir
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", current)
        if folder:
            self.output_edit.setText(folder)
            self.last_output_dir = folder

    def show_selected_preview(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            self.clear_preview_if_empty()
            return

        record = self.records[rows[0]]
        self.selection_info.setText(
            f"已选择 {len(rows)} 张图像；当前：{record.path.name}  |  "
            f"{record.width} × {record.height} px  |  {human_size(record.file_size)}"
        )
        self.preview_meta.setText(
            f"{record.width} × {record.height} px  ·  {record.mode}  ·  "
            f"{record.image_format}  ·  {human_size(record.file_size)}"
        )

        # 使用缓存
        cache_key = str(record.path)
        if cache_key in self._image_cache:
            self.preview_label.setPixmap(self._image_cache[cache_key])
            return

        try:
            with Image.open(record.path) as image:
                image = ImageOps.exif_transpose(image)
                image.seek(0)
                preview = image.convert("RGBA")
                preview.thumbnail((320, 160), Image.Resampling.LANCZOS)
                raw = preview.tobytes("raw", "RGBA")
                qimage = QImage(
                    raw,
                    preview.width,
                    preview.height,
                    preview.width * 4,
                    QImage.Format.Format_RGBA8888,
                ).copy()
                pixmap = QPixmap.fromImage(qimage)
                self._image_cache[cache_key] = pixmap  # 缓存
                self.preview_label.setPixmap(pixmap)
        except Exception:
            self.preview_label.clear()
            self.preview_label.setText("预览不可用")

    def validate_before_process(self) -> tuple[Path, list[str]] | None:
        if not self.records:
            QMessageBox.information(self, "没有图像", "请先选择图像或图像文件夹。")
            return None

        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "缺少输出目录", "请选择输出目录。")
            return None

        names = [self.output_name(row, record) for row, record in enumerate(self.records)]
        if any(not Path(name).stem for name in names):
            QMessageBox.warning(self, "文件名无效", "生成的文件名为空，请检查命名规则。")
            return None
        if len({name.casefold() for name in names}) != len(names):
            QMessageBox.warning(self, "文件名重复", "当前规则产生了重复名称，请调整编号设置。")
            return None

        return Path(output_text).expanduser(), names

    def process_images(self) -> None:
        validated = self.validate_before_process()
        if validated is None:
            return
        output_dir, names = validated

        if self.timestamp_checkbox.isChecked():
            timestamp = get_timestamp()
            output_dir = output_dir / timestamp

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "无法创建输出目录", str(exc))
            return

        # 检查同名文件
        existing = []
        for name in names:
            dest = output_dir / name
            if dest.exists():
                existing.append(name)

        if existing and not self.overwrite_checkbox.isChecked():
            sample = "\n".join(existing[:6])
            more = f"\n……另有 {len(existing) - 6} 个" if len(existing) > 6 else ""
            QMessageBox.warning(
                self,
                "存在同名文件",
                f"输出目录中已有以下文件：\n{sample}{more}\n\n"
                "请更改命名规则、选择其他目录，或勾选允许覆盖。",
            )
            return

        # 清空预览缓存以释放内存
        self._image_cache.clear()

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在处理...")

        self.processor = ImageProcessor(
            self.records,
            names,
            output_dir,
            self.format_combo.currentText(),
            self.overwrite_checkbox.isChecked()
        )
        self.processor.progress.connect(self.update_progress)
        self.processor.status.connect(self.update_status)
        self.processor.finished.connect(self.processing_finished)
        self.processor.error.connect(self.show_error)
        self.processor.start()

    def update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(f"处理进度: {value}%")

    def update_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def processing_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("处理完成!")
        QMessageBox.information(self, "完成", "所有图像处理完成!")

    def show_error(self, error_msg: str) -> None:
        self.statusBar().showMessage(f"错误: {error_msg}")

    def closeEvent(self, event) -> None:
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "处理任务正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processor.stop()
                self.processor.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("图像批量重命名工具")
    app.setStyle("Fusion")
    window = ImageBatchRenamer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
