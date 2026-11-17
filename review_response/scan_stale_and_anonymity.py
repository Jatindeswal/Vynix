#!/usr/bin/env python3
"""
Comprehensive Scanner for Stale Claims and Anonymity Leaks:
Generates:
1. STALE_CLAIMS.md (file:line, text, correct value or action)
2. ANON_RELEASE_CHECKLIST.md (author names, emails, usernames, absolute paths)
"""

import json
import os
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_stale_claims():
    stale_terms = [
        "30.31", "34.80", "36.10", "29.71", "105.0", "54,532", "54532",
        "all 38,118", "all 38118", "38,118", "SOTA", "outperforms",
        "h % 4", "h % 4 == 0", "0.08 threshold", "soft continuous"
    ]

    findings = []
    for root, dirs, files in os.walk(WORKSPACE_DIR):
        if any(ignored in root for ignored in [".git", "__pycache__", "node_modules", "archive"]):
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in [".py", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".bat", ".sh", ".csv", ".ipynb"]:
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, WORKSPACE_DIR).replace("\\", "/")
                # Skip audit files where these numbers are explicitly discussed as audit targets
                if rel_path in ["review_response/FINAL_TABLES.md", "review_response/DISCLOSURES.md", "STALE_CLAIMS.md", "CLEANUP_PLAN.md"]:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for lnum, line in enumerate(f, 1):
                            line_str = line.strip()
                            for term in stale_terms:
                                if term.lower() in line_str.lower():
                                    action = get_suggested_action(term, line_str, rel_path)
                                    findings.append({
                                        "file": rel_path,
                                        "line": lnum,
                                        "term": term,
                                        "text": line_str[:120],
                                        "action": action
                                    })
                except Exception:
                    pass

    return findings


def get_suggested_action(term: str, line: str, rel_path: str) -> str:
    if term in ["30.31", "34.80", "36.10", "29.71"]:
        return "Update to box-level Default mAP (v1.0: 12.23% cont / 12.80% 11P; v2.0: 14.76% cont / 15.29% 11P) or mark OUTDATED (image-level AP)"
    elif term in ["105.0"]:
        return "Correct Vynix 1.0 FPS: measured 2.05-2.20 img/s end-to-end (~455-489 ms/img). 105.0 FPS was adapter-only throughput."
    elif term in ["54,532", "54532"]:
        return "Correct candidate pair count; nominal test hypotheses ~1.08M - 1.11M."
    elif "38,118" in term:
        return "Clarify training shot count: cache sampled max_shots=50 exemplars (total cached: 19,824 for v1.0, 20,006 for v2.0 from ~13.7k-14.8k images)."
    elif term in ["SOTA", "outperforms"]:
        return "Tone down claim: Vynix provides competitive few-shot adaptation; clarify comparisons against fully fine-tuned models under standard box-level mAP."
    elif "h % 4" in term:
        return "Replace non-standard pseudo-rare heuristic with standard HICO-DET 138 Rare / 462 Non-Rare split."
    elif "0.08 threshold" in term:
        return "Clarify effective YOLO confidence threshold was 0.25 in inference due to Ultralytics default."
    elif "soft continuous" in term:
        return "Harmonize description: continuous distance attenuation exp(-(d/sigma)^2) operates on IoU=0 contact interactions."
    return "Audit and correct."


def scan_anonymity():
    anonymity_issues = []

    # 1. Git Log
    try:
        git_log = subprocess.check_output(["git", "log", "--format=%h | %an | %ae | %s"], cwd=WORKSPACE_DIR, text=True)
        authors = set()
        for line in git_log.splitlines():
            parts = line.split(" | ")
            if len(parts) >= 3:
                authors.add((parts[1].strip(), parts[2].strip()))
        for name, email in authors:
            anonymity_issues.append({
                "category": "Git Commit Author/Email",
                "location": "git log",
                "detail": f"Author: '{name}', Email: '{email}'",
                "risk": "High (exposes identity in git history)",
                "action": "Do NOT push commit history to public review repo; prepare anonymous bundle via git archive or fresh initial commit on release branch."
            })
    except Exception as e:
        anonymity_issues.append({"category": "Git Log", "location": "git log", "detail": str(e), "risk": "Low", "action": "Check git"})

    # 2. Check docx files
    docx_files = [f for f in os.listdir(WORKSPACE_DIR) if f.endswith(".docx")]
    for df in docx_files:
        df_path = os.path.join(WORKSPACE_DIR, df)
        try:
            with zipfile.ZipFile(df_path) as z:
                if "docProps/core.xml" in z.namelist():
                    root = ET.fromstring(z.read("docProps/core.xml"))
                    for elem in root.iter():
                        tag = elem.tag.split("}")[-1]
                        if tag in ["creator", "lastModifiedBy", "description"] and elem.text:
                            anonymity_issues.append({
                                "category": "DOCX Metadata",
                                "location": df,
                                "detail": f"{tag}: '{elem.text}'",
                                "risk": "High (metadata leak in document)",
                                "action": "Scrub document core.xml properties before public distribution."
                            })
        except Exception:
            pass

    # 3. Check notebooks
    nb_files = [f for f in os.listdir(WORKSPACE_DIR) if f.endswith(".ipynb")]
    for nb in nb_files:
        nb_path = os.path.join(WORKSPACE_DIR, nb)
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
            meta_str = json.dumps(nb_data.get("metadata", {}))
            abs_paths = re.findall(r'[A-Za-z]:\\[^"\']+', meta_str)
            for p in set(abs_paths):
                anonymity_issues.append({
                    "category": "Jupyter Notebook Metadata",
                    "location": nb,
                    "detail": f"Absolute path in kernel/metadata: '{p}'",
                    "risk": "Medium (leaks OS path / username)",
                    "action": "Clear notebook kernel metadata and execution outputs."
                })
        except Exception:
            pass

    # 4. Check launchers and batch scripts
    for file in os.listdir(WORKSPACE_DIR):
        if file.endswith((".bat", ".vbs", ".sh")):
            fpath = os.path.join(WORKSPACE_DIR, file)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                abs_paths = re.findall(r'[A-Za-z]:\\[^\s"\']+', content)
                for ap in set(abs_paths):
                    anonymity_issues.append({
                        "category": "Launcher Script",
                        "location": file,
                        "detail": f"Hardcoded path: '{ap}'",
                        "risk": "Medium (breaks portability & exposes username)",
                        "action": "Replace hardcoded paths with %~dp0 relative paths or environment variables."
                    })
            except Exception:
                pass

    return anonymity_issues


def generate_reports():
    stale = scan_stale_claims()
    anon = scan_anonymity()

    # 1. Write STALE_CLAIMS.md
    stale_md = "# Project Vynix: Stale Claims and Outdated Metrics Audit (`STALE_CLAIMS.md`)\n\n"
    stale_md += "This document catalogs all occurrences of superseded metrics, non-standard splits, and stale claims across repository files.\n\n"
    stale_md += f"**Total occurrences scanned across repo:** {len(stale)}\n\n"
    stale_md += "| Location (File:Line) | Term | Matched Text Snippet | Required Action / Correct Value |\n"
    stale_md += "| :--- | :--- | :--- | :--- |\n"
    for s in stale:
        clean_text = s['text'].replace('|', '\\|')
        stale_md += f"| `{s['file']}:{s['line']}` | `{s['term']}` | `{clean_text}` | {s['action']} |\n"

    with open(os.path.join(WORKSPACE_DIR, "STALE_CLAIMS.md"), "w", encoding="utf-8") as f:
        f.write(stale_md)
    print(f"✓ Generated STALE_CLAIMS.md with {len(stale)} entries.")

    # 2. Write ANON_RELEASE_CHECKLIST.md
    anon_md = "# Project Vynix: Anonymity and Release Sanitization Checklist (`ANON_RELEASE_CHECKLIST.md`)\n\n"
    anon_md += "This document lists all author identities, email addresses, usernames, and absolute file paths detected across the repository metadata, commit logs, documents, and scripts. **Strictly for local audit; no Git history is rewritten or pushed.**\n\n"
    anon_md += "| Category | Location | Detected Detail | Risk Level | Recommended Release Remediation |\n"
    anon_md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for a in anon:
        clean_detail = a['detail'].replace('|', '\\|')
        anon_md += f"| {a['category']} | `{a['location']}` | `{clean_detail}` | **{a['risk']}** | {a['action']} |\n"

    with open(os.path.join(WORKSPACE_DIR, "ANON_RELEASE_CHECKLIST.md"), "w", encoding="utf-8") as f:
        f.write(anon_md)
    print(f"✓ Generated ANON_RELEASE_CHECKLIST.md with {len(anon)} entries.")


if __name__ == "__main__":
    generate_reports()
