"""Verify and restore the archived study directory tree."""
from pathlib import Path
import hashlib, json, zipfile
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'ARCHIVE_SHA256.json').read_text())
for name, expected in manifest.items():
    archive = root / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Archive checksum mismatch: {name}')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if target.exists() and target.read_bytes() != z.read(item):
                raise FileExistsError(f'Refusing to overwrite modified file: {target}')
        z.extractall(root)
print('Study restored. Run python verify_artifacts.py next.')
