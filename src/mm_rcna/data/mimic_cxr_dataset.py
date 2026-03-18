from __future__ import annotations

from pathlib import Path
import pandas as pd

from mm_rcna.config import AppConfig
from mm_rcna.schemas import ImageRef, StudyRecord
from mm_rcna.data.report_reader import read_text_if_exists


class MIMICCXRStudyBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        p = Path(config.data.studies_csv)
        if p.exists():
            self.df = pd.read_csv(p)
        else:
            self.df = pd.DataFrame([
                {
                    'study_id': 'demo-study-001',
                    'subject_id': 'demo-subject-001',
                    'image_count': 0,
                }
            ])

    def build_study(self, study_id: str) -> StudyRecord:
        row = self.df[self.df['study_id'].astype(str) == str(study_id)]
        if row.empty:
            subject_id = None
        else:
            row = row.iloc[0]
            subject_id = None if 'subject_id' not in row else str(row['subject_id'])
        notes = read_text_if_exists(Path(self.cfg.data.notes_dir) / f'{study_id}.txt')
        report = read_text_if_exists(Path(self.cfg.data.reports_dir) / f'{study_id}.txt')
        images = []
        image_dir = Path(self.cfg.data.images_dir)
        for candidate in sorted(image_dir.glob(f'{study_id}_*')):
            images.append(ImageRef(path=str(candidate), view='unknown'))
        return StudyRecord(
            study_id=str(study_id),
            subject_id=subject_id,
            notes=notes,
            report=report,
            images=images,
        )
