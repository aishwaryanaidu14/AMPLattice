"""Obtain and losslessly convert the checksum-pinned MarLys v3 release."""
import argparse
import csv
import json
import shutil
import urllib.request
from pathlib import Path

from .known_amp_audit import AA, DOI, save_json, sha256

RAW_SHA256 = '298d5e060f27d51714693176068777a0dd77c5ea79a0046d062bbfaef81fb224'
RAW_SIZE = 98103510
RECORD_COUNT = 103200
DOWNLOAD_URL = 'https://data.mendeley.com/public-files/datasets/w4hb5grjwb/files/6c831932-8e0f-4441-9e53-33fae57da262/file_downloaded'


def release_records(raw):
    if sha256(raw) != RAW_SHA256:
        raise ValueError('JSON checksum differs from the pinned MarLys v3 release; refusing to proceed')
    entries = json.loads(Path(raw).read_text())
    if not isinstance(entries, list) or len(entries) != RECORD_COUNT:
        raise ValueError('Unexpected MarLys v3 record count/schema')
    ids = set()
    for record in entries:
        header, seq = record.get('id'), record.get('sequence')
        if not isinstance(header, str) or not header or header in ids or any(c.isspace() for c in header):
            raise ValueError('Invalid or duplicate MarLys entry ID')
        if not isinstance(seq, str) or not seq or set(seq) - AA:
            raise ValueError(f'Noncanonical MarLys record {header!r}; no record will be dropped')
        ids.add(header)
    return entries


def fasta_bytes(entries):
    return ''.join(f">{r['id']}\n{r['sequence']}\n" for r in entries).encode()


def verify_reference_release(fasta, provenance):
    """Verify BOTH original release bytes and a record-complete deterministic conversion."""
    provenance = Path(provenance).resolve()
    data = json.loads(provenance.read_text())
    if data.get('doi') != DOI or data.get('raw_sha256') != RAW_SHA256:
        raise ValueError('Provenance is not the pinned MarLys v3 release')
    raw = provenance.parent/data['raw_file']
    entries = release_records(raw)
    actual = Path(fasta).read_bytes()
    if actual != fasta_bytes(entries):
        raise ValueError('Reference FASTA is not the complete, unfiltered MarLys v3 conversion')
    if sha256(fasta) != data.get('fasta_sha256'):
        raise ValueError('Reference provenance FASTA checksum mismatch')
    return True


def prepare(out, supplied_json=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, fasta, provenance = out/'MLAMP_db.json', out/'marlys_v3.fasta', out/'marlys_v3.provenance.json'
    if supplied_json:
        source = Path(supplied_json).resolve()
        release_records(source)  # Verify before copying or overwriting anything.
        if source != raw:
            if raw.exists() and sha256(raw) != RAW_SHA256:
                raise ValueError('Existing JSON checksum differs; use a new output directory')
            if not raw.exists():
                shutil.copyfile(source, raw)
    elif not raw.exists():
        tmp = raw.with_name(raw.name+'.partial')
        request = urllib.request.Request(DOWNLOAD_URL, headers={'User-Agent': 'amp-known-audit/1'})
        print(f'Downloading MarLys v3 ({RAW_SIZE:,} bytes)...', flush=True)
        with urllib.request.urlopen(request, timeout=60) as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)
        release_records(tmp)  # Partial/wrong downloads never become the cached release.
        tmp.replace(raw)
    entries = release_records(raw)
    converted = fasta_bytes(entries)
    if fasta.exists() and fasta.read_bytes() != converted:
        raise ValueError('Existing FASTA differs; use a new output directory')
    tmp = fasta.with_name(fasta.name+'.tmp')
    tmp.write_bytes(converted)
    tmp.replace(fasta)
    with (out/'marlys_v3.entry_sources.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['entry_id', 'sequence', 'source_databases'])
        writer.writerows([r['id'], r['sequence'], json.dumps(r.get('databases', []))] for r in entries)
    save_json(provenance, dict(doi=DOI, download_url=DOWNLOAD_URL, raw_file=raw.name,
        raw_sha256=RAW_SHA256, raw_size=raw.stat().st_size, fasta_sha256=sha256(fasta),
        records=len(entries), unique_sequences=len({r['sequence'] for r in entries}),
        shortest=min(len(r['sequence']) for r in entries), longest=max(len(r['sequence']) for r in entries),
        conversion='Every original record, unchanged sequence and ID; no deduplication or length filtering'))
    verify_reference_release(fasta, provenance)
    print(f'Verified {len(entries):,} reference records: {fasta}', flush=True)
    return fasta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--json', help='Optional already downloaded MLAMP_db.json; pinned release checksum is mandatory')
    args = p.parse_args()
    prepare(args.out, args.json)


if __name__ == '__main__':
    main()
