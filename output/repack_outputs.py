"""
Repack outputs into GitHub-compliant split zips under 50MB with pure UNIX LF line endings.
Team Gemz - Amazon ML Challenge 2026
"""
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def repack():
    m_tsv = HERE / "matching_results.tsv"
    c_tsv = HERE / "candidate_pairs.tsv"
    
    assert m_tsv.exists(), f"Missing {m_tsv}"
    assert c_tsv.exists(), f"Missing {c_tsv}"
    
    # 1. Verify 0 CRLF bytes in both TSVs
    print("Verifying pure UNIX LF line endings (0 carriage returns)...")
    for p in [m_tsv, c_tsv]:
        with open(p, "rb") as f:
            rc = sum(chunk.count(b"\r") for chunk in iter(lambda: f.read(1024*1024), b""))
        assert rc == 0, f"FATAL: Found {rc} CRLF bytes in {p.name}!"
        print(f"  -> {p.name}: verified 0 \\r bytes.")
        
    # 2. Package matching_results.zip
    m_zip = HERE / "matching_results.zip"
    if m_zip.exists():
        m_zip.unlink()
    print(f"Packaging {m_zip.name}...")
    with zipfile.ZipFile(m_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(m_tsv, arcname="matching_results.tsv")
    print(f"  -> {m_zip.name}: {m_zip.stat().st_size / (1024*1024):.2f} MB")
    
    # 3. Split candidate_pairs.tsv into 4 equal chunks
    print("Splitting candidate_pairs.tsv into 4 parts...")
    with open(c_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    n_lines = len(lines)
    chunk_size = (n_lines + 3) // 4
    
    for i in range(1, 5):
        st = (i - 1) * chunk_size
        en = min(i * chunk_size, n_lines)
        part_name = f"candidate_part_{i}.tsv"
        part_zip = HERE / f"candidate_pairs_part{i}.zip"
        if part_zip.exists():
            part_zip.unlink()
            
        part_text = "".join(lines[st:en])
        with zipfile.ZipFile(part_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr(part_name, part_text)
        print(f"  -> {part_zip.name}: {part_zip.stat().st_size / (1024*1024):.2f} MB ({en - st:,} lines)")
        
    print("\nRepacking complete and verified!")

if __name__ == "__main__":
    repack()
