"""Exhaustive MMseqs2 audit; organizer normalization/settings remain unspecified.

Uses the MMseqs2 guide's fake-prefilter method, not a heuristic search.
This module intentionally has only standard-library dependencies.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

AA = set('ACDEFGHIKLMNPQRSTVWY')
DOI = 'https://doi.org/10.17632/w4hb5grjwb.3'
FIELDS = 'query,target,bits,raw,nident,alnlen,qstart,qend,tstart,tend,qlen,tlen,cigar'
SCHEMA = 1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_records(path, candidate=False):
    """Strict FASTA: no filtering, length truncation, residue replacement or deduplication."""
    records, header, pieces = [], None, []
    def finish():
        if header is None:
            return
        seq = ''.join(pieces).upper()
        if not seq or set(seq) - AA:
            raise ValueError(f'{path}: invalid/empty sequence in {header!r}; not silently discarded')
        if candidate and not 8 <= len(seq) <= 50:
            raise ValueError(f'{path}: candidate {header!r} has length {len(seq)}, outside 8..50')
        records.append((header, seq))
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                finish()
                header, pieces = line[1:].strip(), []
                if not header:
                    raise ValueError(f'{path}: empty FASTA header')
            elif header is None:
                raise ValueError(f'{path}: sequence before first FASTA header')
            else:
                pieces.append(''.join(line.split()))
    finish()
    if not records:
        raise ValueError(f'{path}: empty FASTA')
    if candidate and len({s for _, s in records}) != len(records):
        raise ValueError(f'{path}: duplicate candidate sequences')
    return records


def write_records(path, records, prefix, offset=0):
    with Path(path).open('w') as f:
        for i, (_, seq) in enumerate(records, offset):
            f.write(f'>{prefix}{i:09d}\n{seq}\n')


def fake_prefilter(query_db, target_db, output, diagonal=False):
    """Explicitly nominate EVERY target for EVERY query, or paired self targets."""
    query_db, target_db, output = map(Path, (query_db, target_db, output))
    keys = [line.split()[0] for line in Path(str(query_db) + '.index').read_text().splitlines()]
    if diagonal:
        offset = 0
        with output.open('wb') as data, Path(str(output) + '.index').open('w') as index:
            for key in keys:
                entry = f'{key}\t0\t0\n'.encode() + b'\0'
                data.write(entry)
                index.write(f'{key}\t{offset}\t{len(entry)}\n')
                offset += len(entry)
    else:
        target_index = Path(str(target_db) + '.index').resolve()
        output.symlink_to(target_index)
        size = target_index.stat().st_size
        Path(str(output) + '.index').write_text(''.join(f'{key}\t0\t{size}\n' for key in keys))
    Path(str(output) + '.dbtype').write_bytes(struct.pack('<I', 7))


def alignment_settings(threads, max_len):
    return ['--alignment-mode', '3', '-a', '1', '--seq-id-mode', '0',
            '--comp-bias-corr', '0', '--sub-mat', 'aa:blosum62.out,nucl:nucleotide.out',
            '--gap-open', 'aa:11,nucl:5', '--gap-extend', 'aa:1,nucl:2',
            '--score-bias', '0', '--corr-score-weight', '0', '--realign', '0',
            '--alt-ali', '0', '-e', 'inf', '--min-seq-id', '0', '-c', '0',
            '--min-aln-len', '0', '--max-accept', '2147483647',
            '--max-rejected', '2147483647', '--max-seq-len', str(max_len),
            '--threads', str(threads), '-v', '1']


def run_command(executable, args, log):
    with Path(log).open('a') as f:
        f.write(json.dumps([executable, *map(str, args)]) + '\n')
        f.flush()
        subprocess.run([executable, *map(str, args)], check=True, stdout=f, stderr=f)


def align_pair(executable, query_db, target_db, work, settings, log, diagonal=False):
    work.mkdir()
    pref, result, tsv = work/'pref', work/'result', work/'alignments.tsv'
    fake_prefilter(query_db, target_db, pref, diagonal=diagonal)
    run_command(executable, ['align', query_db, target_db, pref, result, *settings], log)
    run_command(executable, ['convertalis', query_db, target_db, result, tsv,
                            '--format-output', FIELDS, '--threads', settings[-3], '-v', '1'], log)
    return tsv


def parse_alignment(line):
    parts = line.rstrip('\n').split('\t')
    if len(parts) != 13:
        raise ValueError(f'Unexpected MMseqs2 columns: {parts!r}')
    q, t, bits, raw, ni, al, qs, qe, ts, te, ql, tl, cigar = parts
    row = dict(query=q, target=t, bits=float(bits), raw=int(raw), nident=int(ni),
               alnlen=int(al), qstart=int(qs), qend=int(qe), tstart=int(ts),
               tend=int(te), qlen=int(ql), tlen=int(tl), cigar=cigar)
    if not math.isfinite(row['bits']) or row['bits'] < 0:
        raise ValueError('Invalid bit score')
    if row['raw'] <= 0 or row['alnlen'] == 0:
        return row, False  # MMseqs2 may emit a degenerate, zero-score alignment.
    if not (0 <= row['nident'] <= row['alnlen'] and
            1 <= row['qstart'] <= row['qend'] <= row['qlen'] and
            1 <= row['tstart'] <= row['tend'] <= row['tlen']):
        raise ValueError(f'Invalid alignment coordinates/counts: {row}')
    row['identity'] = row['nident'] / row['alnlen']
    row['query_coverage'] = (row['qend'] - row['qstart'] + 1) / row['qlen']
    row['target_coverage'] = (row['tend'] - row['tstart'] + 1) / row['tlen']
    # Integer counts, not rounded fident: EXACTLY 80% is allowed.
    row['above_80_percent'] = 5 * row['nident'] > 4 * row['alnlen']
    return row, True


def new_summary():
    return {'positive_alignments': 0, 'above_80_alignments': 0,
            'above_80_with_query_coverage_80': 0, 'best_bits': None, 'best_identity': None}


def update(summary, row):
    summary['positive_alignments'] += 1
    summary['above_80_alignments'] += int(row['above_80_percent'])
    summary['above_80_with_query_coverage_80'] += int(row['above_80_percent'] and
        5 * (row['qend'] - row['qstart'] + 1) >= 4 * row['qlen'])
    for name, rank in [('best_bits', lambda r: (r['bits'], r['raw'], r['identity'], r['target'])),
                       ('best_identity', lambda r: (r['identity'], r['query_coverage'], r['bits'], r['target']))]:
        if summary[name] is None or rank(row) > rank(summary[name]):
            summary[name] = row


def merge(total, part):
    for name in ('positive_alignments', 'above_80_alignments', 'above_80_with_query_coverage_80'):
        total[name] += part[name]
    for name, rank in [('best_bits', lambda r: (r['bits'], r['raw'], r['identity'], r['target'])),
                       ('best_identity', lambda r: (r['identity'], r['query_coverage'], r['bits'], r['target']))]:
        row = part[name]
        if row is not None and (total[name] is None or rank(row) > rank(total[name])):
            total[name] = row


def _audit(args):
    if args.threads < 1 or args.query_batch < 1 or args.reference_batch < 1:
        raise ValueError('threads and batch sizes must be positive')
    library = read_records(args.library, candidate=True)
    top = read_records(args.top, candidate=True) if args.top else []
    if not top and args.scope != 'library':
        raise ValueError('--scope top requires --top; library-only alignment may omit it')
    if top and len(top) != 100:
        raise ValueError(f'--top must contain exactly 100 unique sequences, got {len(top)}')
    library_set = {s for _, s in library}
    if any(s not in library_set for _, s in top):
        raise ValueError('Top-100 is not a subset of this library. Supply its matching top FASTA.')
    reference = read_records(args.reference)
    actual_sha = sha256(args.reference)
    if args.reference_sha256 and actual_sha != args.reference_sha256.lower():
        raise ValueError('Reference SHA256 mismatch')
    verified_release = False
    provenance = getattr(args, 'reference_provenance', None)
    if provenance:
        from .marlys_reference import verify_reference_release
        verified_release = verify_reference_release(args.reference, provenance)
    executable = shutil.which(args.mmseqs)
    if executable is None:
        raise RuntimeError('MMseqs2 is missing; install it or provide --mmseqs /absolute/path')
    executable = str(Path(executable).resolve())
    version = subprocess.check_output([executable, 'version'], text=True).strip()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    max_len = max(65535, max(len(s) for _, s in reference))
    settings = alignment_settings(args.threads, max_len)
    manifest = dict(schema=SCHEMA, reference_sha256=actual_sha,
                    reference_source=args.reference_source, claimed_reference_release=DOI,
                    reference_release_independently_verified=verified_release,
                    library_sha256=sha256(args.library), top_sha256=sha256(args.top) if args.top else None,
                    reference_records=len(reference), reference_unique_sequences=len({s for _, s in reference}),
                    library_records=len(library), top_records=len(top),
                    mmseqs_version=version, mmseqs_binary_sha256=sha256(executable),
                    settings=settings, fields=FIELDS, query_batch=args.query_batch,
                    reference_batch=args.reference_batch, scope=args.scope,
                    keep_alignments=args.keep_alignments,
                    organizer_score_reproduced=False,
                    unresolved=['organizer bit-score normalization and aggregation',
                                'organizer coverage, significance and scoring settings'],
                    pair_nomination='exhaustive fake prefilter: all records, no search, no cap',
                    no_positive_alignment='No positive local alignment returned by exhaustive align; not a heuristic search miss')
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output manifest differs from this run; use a new --out directory')
    if not manifest_path.exists():
        if any(p.name != '.audit.lock' for p in out.iterdir()):
            raise ValueError('Output directory has files but no manifest; use a new directory')
        save_json(manifest_path, manifest)
    # Always write an immediate, exact library-wide MarLys overlap audit.
    known = {}
    for i, (header, seq) in enumerate(reference):
        known.setdefault(seq, []).append((i, header))
    exact_rows = [(i, header, seq, known[seq]) for i, (header, seq) in enumerate(library) if seq in known]
    with (out/'exact_matches.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['library_index', 'library_header', 'sequence', 'reference_indices', 'reference_headers'])
        for i, header, seq, hits in exact_rows:
            writer.writerow([i, header, seq, json.dumps([j for j, _ in hits]), json.dumps([h for _, h in hits])])
    save_json(out/'exact_summary.json', dict(library_size=len(library), exact_matches=len(exact_rows),
              exact_match_fraction=len(exact_rows)/len(library),
              fraction_absent_from_reference=1-len(exact_rows)/len(library),
              reference_sha256=actual_sha, reference_records=len(reference)))
    # Put the top100 first even for the expensive library-wide alignment run.
    top_set = {s for _, s in top}
    queries = top + ([r for r in library if r[1] not in top_set] if args.scope == 'library' else [])
    print(f'Exact MarLys overlap: {len(exact_rows)}/{len(library)}', flush=True)
    print(f'Exhaustive alignment: {len(queries):,} x {len(reference):,} = {len(queries)*len(reference):,} pairs', flush=True)
    log = out/'commands.log'
    # Mapping preserves all original records, including duplicate reference entries.
    for name, records, prefix in [('query', queries, 'q'), ('reference', reference, 'r')]:
        with (out/f'{name}_mapping.tsv').open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['audit_id', 'original_header', 'sequence'])
            writer.writerows((f'{prefix}{i:09d}', h, s) for i, (h, s) in enumerate(records))
    refs_dir = out/'reference_databases'
    refs_dir.mkdir(exist_ok=True)
    target_dbs = []
    for start in range(0, len(reference), args.reference_batch):
        folder = refs_dir/f'{start:09d}'
        marker = folder/'complete.json'
        if not marker.exists():
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()
            write_records(folder/'reference.fasta', reference[start:start+args.reference_batch], 'r', start)
            run_command(executable, ['createdb', folder/'reference.fasta', folder/'db', '--dbtype', '1', '--shuffle', '0', '-v', '1'], log)
            save_json(marker, {'start': start, 'count': min(args.reference_batch, len(reference)-start)})
        target_dbs.append((start, folder/'db'))
    batches = out/'batches'
    batches.mkdir(exist_ok=True)
    results = []
    for start in range(0, len(queries), args.query_batch):
        batch = batches/f'{start:09d}'
        batch.mkdir(exist_ok=True)
        chunk = queries[start:start+args.query_batch]
        ids = [f'q{i:09d}' for i in range(start, start+len(chunk))]
        summaries = {q: new_summary() for q in ids}
        qwork = batch/'query_work'
        if qwork.exists():
            shutil.rmtree(qwork)
        qwork.mkdir()
        write_records(qwork/'query.fasta', chunk, 'q', start)
        qdb = qwork/'db'
        run_command(executable, ['createdb', qwork/'query.fasta', qdb, '--dbtype', '1', '--shuffle', '0', '-v', '1'], log)
        self_path = batch/'self_scores.json'
        if self_path.exists():
            self_scores = json.loads(self_path.read_text())
        else:
            tsv = align_pair(executable, qdb, qdb, qwork/'self', settings, log, diagonal=True)
            self_scores = {}
            for line in tsv.open():
                row, valid = parse_alignment(line)
                if not valid or row['query'] != row['target'] or row['bits'] <= 0:
                    raise RuntimeError('Invalid diagonal self alignment; cannot normalize')
                self_scores[row['query']] = row['bits']
            if set(self_scores) != set(ids):
                raise RuntimeError('Missing self alignments; cannot normalize')
            save_json(self_path, self_scores)
        for rstart, target_db in target_dbs:
            done = batch/f'reference_{rstart:09d}.json'
            if done.exists():
                part = json.loads(done.read_text())
            else:
                work = qwork/'alignment'
                if work.exists():
                    shutil.rmtree(work)
                tsv = align_pair(executable, qdb, target_db, work, settings, log)
                part = {q: new_summary() for q in ids}
                seen = set()
                rstop = min(rstart+args.reference_batch, len(reference))
                with tsv.open() as f:
                    for line in f:
                        row, valid = parse_alignment(line)
                        q, t = row['query'], row['target']
                        if q not in part or not t.startswith('r') or not rstart <= int(t[1:]) < rstop:
                            raise RuntimeError('Unexpected alignment identifier')
                        if (q, t) in seen:
                            raise RuntimeError('Duplicate pair output; audit expects one primary alignment per pair')
                        seen.add((q, t))
                        if valid:
                            if row['qlen'] != len(queries[int(q[1:])][1]) or row['tlen'] != len(reference[int(t[1:])][1]):
                                raise RuntimeError('Sequence length differs from source; possible truncation')
                            update(part[q], row)
                if args.keep_alignments:
                    raw_path = batch/f'reference_{rstart:09d}.tsv.gz'
                    tmp = raw_path.with_name(raw_path.name+'.tmp')
                    with tsv.open('rb') as src, gzip.open(tmp, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    tmp.replace(raw_path)
                save_json(done, part)  # Atomic marker only after successful parsing/storage.
                shutil.rmtree(work)
            if set(part) != set(ids):
                raise RuntimeError('Cached alignment chunk has missing query IDs')
            for q in ids:
                merge(summaries[q], part[q])
            print(f'Queries {start+1}-{start+len(chunk)}/{len(queries)}; reference {min(rstart+args.reference_batch,len(reference))}/{len(reference)}', flush=True)
        for offset, (header, seq) in enumerate(chunk):
            q = ids[offset]
            item = dict(audit_id=q, original_header=header, sequence=seq, is_top100=seq in top_set,
                        exact_reference_match=seq in known, reference_pairs_nominated=len(reference),
                        self_bits=self_scores[q], **summaries[q])
            best = item['best_bits']
            item['diagnostic_max_bits_over_query_self_bits'] = None if best is None else best['bits']/self_scores[q]
            item['identity_gate_under_declared_settings'] = 'fail' if item['above_80_alignments'] else 'pass'
            if seq in known and not item['above_80_alignments']:
                raise RuntimeError('Exact reference match was not detected by alignment; refusing a false pass')
            results.append(item)
        save_json(batch/'query_summary.json', results[-len(chunk):])
        shutil.rmtree(qwork)
        # Useful checkpoint even during a library-wide run, once all100 have finished.
        if top and len(results) >= 100:
            save_json(out/'top100_alignment_details.json', results[:100])
    save_json(out/'alignment_details.json', results)
    with (out/'per_sequence.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['audit_id', 'sequence', 'is_top100', 'exact_match', 'self_bits',
                         'max_bits', 'max_bits_reference', 'diagnostic_bits_over_query_self',
                         'max_identity', 'max_identity_reference', 'identity_alnlen',
                         'identity_query_coverage', 'identity_target_coverage', 'above80_reference_count',
                         'above80_querycoverage80_count', 'gate_under_declared_settings'])
        for item in results:
            b, i = item['best_bits'], item['best_identity']
            writer.writerow([item['audit_id'], item['sequence'], item['is_top100'], item['exact_reference_match'],
                item['self_bits'], b['bits'] if b else '', b['target'] if b else '',
                item['diagnostic_max_bits_over_query_self_bits'], i['identity'] if i else '',
                i['target'] if i else '', i['alnlen'] if i else '', i['query_coverage'] if i else '',
                i['target_coverage'] if i else '', item['above_80_alignments'],
                item['above_80_with_query_coverage_80'], item['identity_gate_under_declared_settings']])
    top_results = results[:100] if top else []
    report = dict(status='complete_under_declared_protocol', organizer_score_reproduced=False,
                  reference_release_independently_verified=verified_release, scope=args.scope,
                  exact_library_matches=len(exact_rows), library_size=len(library),
                  aligned_queries=len(results), exhaustive_pairs_nominated=len(results)*len(reference),
                  top100_above80_count=sum(r['above_80_alignments'] > 0 for r in top_results) if top else None,
                  top100_exact_matches=sum(r['exact_reference_match'] for r in top_results) if top else None,
                  top100_gate_under_declared_settings=all(r['above_80_alignments'] == 0 for r in top_results) if top else None,
                  caveat='This is an exhaustive declared-protocol audit, not certification against unpublished organizer parameters. Query-self normalized bits are diagnostic only. No property/MIC/selection scores were changed.')
    save_json(out/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def audit(args):
    # WSL/Linux: prevent two processes from mutating the same resumable run.
    import fcntl
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.audit.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another audit is already using this output directory') from exc
        return _audit(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--top', help='Matching top100 FASTA; required for --scope top, optional for library-only alignment')
    p.add_argument('--reference', required=True, help='Full MarLys v3 canonical FASTA, all lengths; never pilot-only known FASTA')
    p.add_argument('--reference-source', required=True, help='Downloaded artifact URL / provenance; saved as supplied, not independently verified')
    p.add_argument('--reference-sha256', help='Expected SHA256 for supplied reference FASTA')
    p.add_argument('--reference-provenance', help='marlys_v3.provenance.json from amp-marlys-reference: validates the original release AND every converted record')
    p.add_argument('--out', required=True)
    p.add_argument('--scope', choices=['top', 'library'], default='top', help='Both scopes check exact overlap across the FULL library; library also exhaustively aligns all library sequences')
    p.add_argument('--mmseqs', default='mmseqs')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--query-batch', type=int, default=64)
    p.add_argument('--reference-batch', type=int, default=8192)
    p.add_argument('--keep-alignments', action='store_true', help='Retain all emitted pair rows compressed; library scope can need substantial disk')
    args = p.parse_args()
    audit(args)


if __name__ == '__main__':
    main()
