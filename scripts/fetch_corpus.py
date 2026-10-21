"""Download the scikit-learn user guide (reStructuredText) pinned to a release tag.

Licence: scikit-learn is BSD-3-Clause. Redistribution requires keeping the copyright
notice and licence text, so we download at build time rather than committing the text,
and we write the licence next to the data.
"""
import sys, urllib.request, pathlib

TAG = "1.5.2"
BASE = f"https://raw.githubusercontent.com/scikit-learn/scikit-learn/{TAG}"
DOCS = """clustering cross_validation model_evaluation ensemble linear_model svm tree preprocessing
feature_selection neighbors naive_bayes impute decomposition calibration compose grid_search
outlier_detection random_projection neural_networks_supervised manifold mixture""".split()

def main(out="data/raw"):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    for name in DOCS:
        dest = out / f"{name}.rst"
        if dest.exists():
            continue
        url = f"{BASE}/doc/modules/{name}.rst"
        print("fetch", url)
        urllib.request.urlretrieve(url, dest)
    urllib.request.urlretrieve(f"{BASE}/COPYING", out / "LICENSE-scikit-learn.txt")
    (out / "SOURCE.txt").write_text(
        f"scikit-learn {TAG} user guide, doc/modules/*.rst\nhttps://github.com/scikit-learn/scikit-learn/tree/{TAG}/doc/modules\n"
        "Licence: BSD-3-Clause (see LICENSE-scikit-learn.txt)\n")
    print("done:", len(DOCS), "documents")

if __name__ == "__main__":
    main(*sys.argv[1:])
