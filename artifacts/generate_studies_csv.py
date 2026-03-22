from pathlib import Path
import csv

raw_root = Path("/home/azureuser/mimic-cxr")
out_csv = Path("/home/yicheng/mm_rcna_mimic/artifacts/studies.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for pfx in sorted(raw_root.glob("p*")):
    if not pfx.is_dir():
        continue
    for pdir in sorted(pfx.glob("p*")):
        if not pdir.is_dir():
            continue
        subject_id = pdir.name[1:] if pdir.name.startswith("p") else pdir.name
        for sdir in sorted(pdir.glob("s*")):
            if not sdir.is_dir():
                continue
            study_id = sdir.name[1:] if sdir.name.startswith("s") else sdir.name
            rows.append((study_id, subject_id))

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["study_id", "subject_id"])
    writer.writerows(rows)

print(f"saved {len(rows)} rows to {out_csv}")