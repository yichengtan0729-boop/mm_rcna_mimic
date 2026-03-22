from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from mm_rcna.config import AppConfig
from mm_rcna.schemas import ImageRef, StudyRecord
from mm_rcna.data.report_reader import read_text_if_exists


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}


def _clean_id(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


class MIMICCXRStudyBuilder:
    """
    Goal:
    1) Prefer raw/original MIMIC-style hierarchy whenever possible.
    2) Fall back to the old artifacts layout if present.
    3) Avoid requiring pre-flattened images or per-study copied txt files.

    Supported raw image hierarchy:
      <images_dir>/p10/p10000032/s50414267/*.jpg

    Supported fallback flat hierarchy:
      <images_dir>/50414267_0.jpg
      <images_dir>/50414267_1.jpg

    Supported report/note lookup order:
      1) raw hierarchy under reports_dir / notes_dir
      2) flat <dir>/<study_id>.txt
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

        # Keep compatibility with the repo, but do not REQUIRE a full studies.csv.
        # If it exists, we use it to get subject_id exactly.
        p = Path(config.data.studies_csv)
        if p.exists():
            try:
                self.df = pd.read_csv(p)
            except Exception:
                self.df = pd.DataFrame(columns=["study_id", "subject_id"])
        else:
            self.df = pd.DataFrame(columns=["study_id", "subject_id"])

    def _get_subject_id_from_df(self, study_id: str) -> Optional[str]:
        if self.df.empty or "study_id" not in self.df.columns:
            return None

        row = self.df[self.df["study_id"].astype(str) == str(study_id)]
        if row.empty:
            return None

        row0 = row.iloc[0]
        if "subject_id" not in row0:
            return None

        return _clean_id(row0["subject_id"])

    def _infer_subject_id_from_raw_tree(self, image_root: Path, study_id: str) -> Optional[str]:
        """
        Search raw MIMIC hierarchy:
          pXX/pXXXXXXXX/sYYYYYYYY
        and infer subject_id from parent directory name pXXXXXXXX.
        """
        study_dir_name = f"s{study_id}"
        for study_dir in image_root.glob(f"p*/p*/{study_dir_name}"):
            if not study_dir.is_dir():
                continue
            patient_dir = study_dir.parent.name  # p10000032
            if patient_dir.startswith("p") and len(patient_dir) > 1:
                return patient_dir[1:]
        return None

    def _list_flat_images(self, image_root: Path, study_id: str) -> List[Path]:
        out: List[Path] = []
        for candidate in sorted(image_root.glob(f"{study_id}_*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                out.append(candidate)
        return out

    def _list_raw_images(self, image_root: Path, study_id: str, subject_id: Optional[str]) -> List[Path]:
        """
        Prefer exact raw path if subject_id is known:
          <root>/p10/p10000032/s50414267/*.jpg

        Otherwise fall back to recursive-ish glob by study_id.
        """
        out: List[Path] = []
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        # Exact path first
        if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
            exact_dir = image_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}"
            if exact_dir.exists() and exact_dir.is_dir():
                for candidate in sorted(exact_dir.iterdir()):
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                        out.append(candidate)
                if out:
                    return out

        # Fallback: search by study directory
        for candidate in sorted(image_root.glob(f"p*/p*/s{study_id}/*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                out.append(candidate)

        return out

    def _read_flat_txt(self, root: Path, study_id: str) -> str:
        return read_text_if_exists(root / f"{study_id}.txt")

    def _read_raw_txt(self, root: Path, study_id: str, subject_id: Optional[str]) -> str:
        """
        Try a few raw-style patterns under root.

        We do not assume one exact report tree, so try:
          <root>/p10/p10000032/s50414267.txt
          <root>/p10/p10000032/50414267.txt
          <root>/p10/p10000032/s50414267/*.txt
        """
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
            pfx = f"p{subject_id[:2]}"
            pdir = f"p{subject_id}"

            direct_candidates = [
                root / pfx / pdir / f"s{study_id}.txt",
                root / pfx / pdir / f"{study_id}.txt",
            ]
            for p in direct_candidates:
                txt = read_text_if_exists(p)
                if txt:
                    return txt

            nested_dir = root / pfx / pdir / f"s{study_id}"
            if nested_dir.exists() and nested_dir.is_dir():
                for p in sorted(nested_dir.glob("*.txt")):
                    txt = read_text_if_exists(p)
                    if txt:
                        return txt

        # fallback by study id only
        for p in sorted(root.glob(f"p*/p*/s{study_id}.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt

        for p in sorted(root.glob(f"p*/p*/{study_id}.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt

        for p in sorted(root.glob(f"p*/p*/s{study_id}/*.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt

        return ""

    def _load_notes(self, study_id: str, subject_id: Optional[str]) -> str:
        root = Path(self.cfg.data.notes_dir)

        # Prefer raw/original-style first
        txt = self._read_raw_txt(root, study_id, subject_id)
        if txt:
            return txt

        # Fall back to old artifacts layout
        return self._read_flat_txt(root, study_id)

    def _load_report(self, study_id: str, subject_id: Optional[str]) -> str:
        root = Path(self.cfg.data.reports_dir)

        # Prefer raw/original-style first
        txt = self._read_raw_txt(root, study_id, subject_id)
        if txt:
            return txt

        # Fall back to old artifacts layout
        return self._read_flat_txt(root, study_id)

    def build_study(self, study_id: str) -> StudyRecord:
        study_id = _clean_id(study_id)
        image_root = Path(self.cfg.data.images_dir)

        # subject_id priority:
        # 1) studies.csv
        # 2) infer from raw hierarchy
        subject_id = self._get_subject_id_from_df(study_id)
        if not subject_id:
            subject_id = self._infer_subject_id_from_raw_tree(image_root, study_id)

        # Prefer raw images first, then flat fallback
        raw_images = self._list_raw_images(image_root, study_id, subject_id)
        flat_images = self._list_flat_images(image_root, study_id) if not raw_images else []
        final_images = raw_images if raw_images else flat_images

        notes = self._load_notes(study_id, subject_id)
        report = self._load_report(study_id, subject_id)

        images = [ImageRef(path=str(p), view="unknown") for p in final_images]

        return StudyRecord(
            study_id=str(study_id),
            subject_id=subject_id,
            notes=notes,
            report=report,
            images=images,
        )