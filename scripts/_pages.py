import json, re, subprocess, sys
from pathlib import Path
S = Path("/tmp/claude-0/-home-claude/1a07952f-25d8-5a39-9e10-b1737da07fc5/scratchpad")
pdf = sys.argv[1]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
anchors = json.loads((S / "anchors.json").read_text())
res = {}
for a in anchors:
    cap = re.match(r"(Figure|Table) (\d+):", a)
    for i, txt in enumerate(pages, 1):
        if cap and i < 11: continue
        lines = [l.strip() for l in txt.splitlines()]
        if cap:
            hit = any(l.startswith(f"{cap.group(1)} {cap.group(2)}:") and "...." not in l for l in lines)
        else:
            hit = any(l == a for l in lines)
        if hit:
            res[a] = str(i); break
    else:
        print("NOT FOUND:", a)
(S / "pages.json").write_text(json.dumps(res))
print(n, "pages;", len(res), "anchors")
