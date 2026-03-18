from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mm_rcna.config import load_config
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.utils.io_utils import save_pickle, ensure_dir


def build_index(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_dir(cfg.data.index_dir)
    studies = pd.read_csv(cfg.data.studies_csv)
    labels = pd.read_csv(cfg.data.labels_csv) if Path(cfg.data.labels_csv).exists() else pd.DataFrame(columns=['study_id'])
    label_map = labels.set_index('study_id').to_dict(orient='index') if not labels.empty else {}

    text_encoder = TextEncoder(cfg.models.text_encoder_name)
    vision = VisionToolRunner(
        cfg.models.image_encoder_name,
        cfg.models.vision_backbone_name,
        cfg.models.lesion_labels,
        cfg.models.lung_regions,
        cfg.models.image_size,
        device='cpu',
    )

    metas = []
    text_vecs = []
    image_vecs = []
    for _, row in studies.iterrows():
        sid = str(row['study_id'])
        subject_id = None if 'subject_id' not in row else str(row['subject_id'])
        note_path = Path(cfg.data.notes_dir) / f'{sid}.txt'
        report_path = Path(cfg.data.reports_dir) / f'{sid}.txt'
        text = ''
        if note_path.exists():
            text += note_path.read_text(encoding='utf-8', errors='ignore') + ' '
        if report_path.exists():
            text += report_path.read_text(encoding='utf-8', errors='ignore')
        tvec = text_encoder.encode([text])[0]
        image_paths = [str(p) for p in Path(cfg.data.images_dir).glob(f'{sid}_*')]
        vout = vision.run(image_paths)
        ivec = np.asarray(vout.feature_vector, dtype=float)
        text_vecs.append(tvec)
        image_vecs.append(ivec)
        metas.append({
            'study_id': sid,
            'subject_id': subject_id,
            'labels': label_map.get(sid, {}),
        })

    np.save(Path(cfg.data.index_dir) / 'text.npy', np.asarray(text_vecs, dtype=np.float32))
    np.save(Path(cfg.data.index_dir) / 'image.npy', np.asarray(image_vecs, dtype=np.float32))
    save_pickle(metas, Path(cfg.data.index_dir) / 'meta.pkl')
    print(f'Index saved to {cfg.data.index_dir}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    build_index(args.config)


if __name__ == '__main__':
    main()
