# ===== Add-on: recover Bengali letter names (run AFTER the main notebook has finished) =====
MERGED_SLUG = "mdnaimurmahi/bn-en-sign-language"   # <-- put your co-author's Kaggle username here
import os, json, zipfile, glob, numpy as np, pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
OUT = "/content/drive/MyDrive/signlink_rerun" if os.path.exists("/content/drive") else os.path.abspath("signlink_rerun")
DATA = "/content/data" if os.path.exists("/content") else os.path.abspath("data")
MERGED = f"{DATA}/merged"
if not os.path.exists(MERGED):
    os.system(f"kaggle datasets download -d {MERGED_SLUG} -p {DATA} -q && unzip -q -o {DATA}/{MERGED_SLUG.split('/')[1]}.zip -d {MERGED}")
# the merged folder is the one with 75 class sub-folders
root = next(dp for dp, dns, _ in os.walk(MERGED) if len(dns) == 75)
z = np.load(f"{OUT}/images_50.npz", allow_pickle=True)
X, meta = z["X"], pd.DataFrame(z["meta"].tolist(), columns=["path", "label", "lang"])
def dhash_arr(x):
    g = np.asarray(Image.fromarray(x).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0]
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
bn_codes = sorted(meta.label[meta.lang == "Bengali"].unique())
en_names = {l[3:] for l in meta.label[meta.lang == "English"].unique()}
named = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in en_names)
print(len(bn_codes), "numbered Bengali classes;", len(named), "named Bengali folders in the merged dataset")
H_code = {c: np.array([dhash_arr(x) for x in X[(meta.label == c).values][:300]], dtype=np.uint64) for c in bn_codes}
def load50(p): return np.asarray(Image.open(p).convert("RGB").resize((50, 50), Image.BILINEAR), dtype=np.uint8)
H_name = {n: np.array([dhash_arr(load50(p)) for p in sorted(glob.glob(os.path.join(root, n, "*")))[:600]], dtype=np.uint64) for n in named}
score = np.zeros((len(bn_codes), len(named)))
for i, c in enumerate(bn_codes):
    for j, n in enumerate(named):
        d = POP[(H_code[c][:, None] ^ H_name[n][None, :]).view(np.uint8).reshape(len(H_code[c]), len(H_name[n]), 8)].sum(-1)
        score[i, j] = (d.min(1) <= 4).mean()          # share of this class's images with a near-identical image in folder n
r, c = linear_sum_assignment(-score)
mapping = pd.DataFrame({"code": [bn_codes[i] for i in r], "letter": [named[j] for j in c], "match_share": score[r, c].round(3),
                        "runner_up_share": [np.sort(score[i])[-2].round(3) for i in r]})
mapping.to_csv(f"{OUT}/tables/bengali_class_names.csv", index=False)
print(mapping.to_string(index=False))
print("\nLow-confidence rows (match_share < 0.5):", int((mapping.match_share < 0.5).sum()))
with zipfile.ZipFile("signlink_results.zip", "a") as zf:
    zf.write(f"{OUT}/tables/bengali_class_names.csv", "tables/bengali_class_names.csv")
print("Added to signlink_results.zip")
