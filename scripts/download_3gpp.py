import os
import re
import sys
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# A diverse, representative spec set spanning RAN, PHY/MAC, 5G core, NAS, security.
DEFAULT_SPECS = ["38.331", "38.300", "38.321", "23.501", "24.501", "33.501"]


def etsi_number(spec: str) -> str:
    """'38.331' -> '138331' (ETSI prefixes 3GPP numbers with a leading 1)."""
    return "1" + spec.replace(".", "")


def series_folder(etsi_num: str) -> str:
    base = (int(etsi_num) // 100) * 100
    return f"{base}_{base + 99}"


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_version_folder(etsi_num: str) -> str:
    base_url = (f"https://www.etsi.org/deliver/etsi_ts/"
                f"{series_folder(etsi_num)}/{etsi_num}/")
    html = _fetch(base_url, timeout=40).decode("utf-8", "ignore")
    versions = re.findall(rf'{etsi_num}/(\d+\.\d+\.\d+_\d+)/', html)
    if not versions:
        raise RuntimeError(f"No versions found at {base_url}")
    # sort by numeric version tuple
    def key(v):
        return tuple(int(x) for x in v.split("_")[0].split("."))
    return sorted(set(versions), key=key)[-1]


def download_spec(spec: str) -> str:
    etsi_num = etsi_number(spec)
    ver = latest_version_folder(etsi_num)
    vnum = "".join(f"{int(x):02d}" for x in ver.split("_")[0].split("."))
    url = (f"https://www.etsi.org/deliver/etsi_ts/{series_folder(etsi_num)}/"
           f"{etsi_num}/{ver}/ts_{etsi_num}v{vnum}p.pdf")
    out = os.path.join(RAW_DIR, f"ts_{spec.replace('.', '')}.pdf")
    if os.path.exists(out) and os.path.getsize(out) > 100_000:
        print(f"  [skip] {spec} already present ({os.path.getsize(out)//1024} KB)")
        return out
    print(f"  [get ] TS {spec}  v{ver.split('_')[0]}  <- {url}")
    data = _fetch(url, timeout=180)
    with open(out, "wb") as f:
        f.write(data)
    print(f"  [ok  ] {spec}: {len(data)//1024} KB")
    return out


def main(specs):
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading {len(specs)} real 3GPP specs from ETSI into {RAW_DIR}")
    ok = 0
    for spec in specs:
        try:
            download_spec(spec)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {spec}: {type(e).__name__}: {e}")
    print(f"Done: {ok}/{len(specs)} specs available.")


if __name__ == "__main__":
    specs = sys.argv[1:] or DEFAULT_SPECS
    main(specs)
