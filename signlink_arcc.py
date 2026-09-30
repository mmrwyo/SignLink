# SignLink re-run for IEEE CSDE 2026 (ieee-csde_3826): script version of the Colab notebook, for ARCC MedicineBow.
# Resumes automatically: models whose weights and history already exist in --out are skipped.
import os, sys, json, time, glob, hashlib, zipfile, random, platform
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
import argparse
ap = argparse.ArgumentParser(description="SignLink re-run (IEEE CSDE 2026, ieee-csde_3826) for ARCC or any GPU machine")
ap.add_argument("--out", default=os.environ.get("SIGNLINK_OUT", os.path.abspath("signlink_rerun")), help="results folder (the downloaded Google Drive signlink_rerun folder)")
ap.add_argument("--data", default=os.environ.get("SIGNLINK_DATA", os.path.abspath("data")), help="raw Kaggle data (only needed if images_50.npz is missing)")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--no-pretrained", action="store_true", help="testing only")
ap.add_argument("--models", default="", help="comma-separated subset (testing only)")
ap.add_argument("--fusion-seeds", type=int, default=5)
args = ap.parse_args()
CFG = dict(
    SEED=42, IMG_STORE=50,            # images are stored at 50x50 (as in the paper) and resized inside each model
    IMG_SIZE=224,                      # model input size used in the paper
    BATCH=64, EPOCHS=12, PATIENCE=3,
    LR_PRETRAINED=1e-4, LR_SCRATCH=3e-4,
    SPLIT=(0.64, 0.16, 0.20),          # train / validation / test, as in the paper
    FUSION_SEEDS=5, PRETRAINED=True,
)
OUT, DATA = args.out, args.data
CFG.update(EPOCHS=args.epochs, IMG_SIZE=args.img_size, PRETRAINED=not args.no_pretrained, FUSION_SEEDS=args.fusion_seeds)
for d in [OUT, f"{OUT}/weights", f"{OUT}/histories", f"{OUT}/preds", f"{OUT}/figures", f"{OUT}/tables", DATA]:
    os.makedirs(d, exist_ok=True)
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
print(json.dumps(CFG, indent=1)); print("Output folder:", OUT)

cache = f"{OUT}/images_50.npz"
if not os.path.exists(cache):   # raw data are only needed when the image cache is missing
    for slug, name in [("muntakimrafi/bengali-sign-language-dataset", "bengali"),
                       ("ahmedkhanak1995/sign-language-gesture-images-dataset", "english")]:
        dest = f"{DATA}/{name}"
        if not os.path.exists(dest):
            os.system(f"kaggle datasets download -d {slug} -p {DATA} -q && unzip -q -o {DATA}/{slug.split('/')[1]}.zip -d {dest}")

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def class_dirs(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        subs = []
        for d in dns:
            p = os.path.join(dp, d)
            n = sum(1 for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
            if n > 0: subs.append((d, n))
        if len(subs) >= 10: out[dp] = subs
    return out

def pick_root(root, n_classes, expected):
    cands = {k: v for k, v in class_dirs(root).items() if len(v) == n_classes}
    print(f"\nCandidates under {root} with {n_classes} class folders:")
    for k, v in cands.items(): print(f"  {k}: {sum(n for _, n in v)} images")
    assert cands, f"No folder with {n_classes} class sub-folders found under {root}; set the root manually."
    best = min(cands, key=lambda k: abs(sum(n for _, n in cands[k]) - expected))
    keys = list(cands)
    for i in range(len(keys)):                       # dataset shipped as two parts (e.g. train/test)?
        for j in range(i + 1, len(keys)):
            tot = sum(n for _, n in cands[keys[i]]) + sum(n for _, n in cands[keys[j]])
            if abs(tot - expected) < abs(sum(n for _, n in cands[best]) - expected) and abs(tot - expected) / expected < 0.02:
                print("  -> using two parts:", keys[i], keys[j]); return [keys[i], keys[j]]
    print("  -> using:", best); return [best]

if os.path.exists(cache):
    EN_ROOTS = BN_ROOTS = ["loaded from images_50.npz (built in the Colab session)"]
else:
    EN_ROOTS = pick_root(f"{DATA}/english", 37, 55500)
    BN_ROOTS = pick_root(f"{DATA}/bengali", 38, 12581)

def load_one(path):
    im = Image.open(path).convert("RGB").resize((CFG["IMG_STORE"], CFG["IMG_STORE"]), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

def collect(roots, prefix, lang):
    rows = []
    for r in roots:
        for d in sorted(os.listdir(r)):
            p = os.path.join(r, d)
            if not os.path.isdir(p): continue
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(IMG_EXT): rows.append((os.path.join(p, f), f"{prefix}_{d}", lang))
    return rows

if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); X_all, meta = z["X"], pd.DataFrame(z["meta"].tolist(), columns=["path", "label", "lang"])
else:
    rows = collect(EN_ROOTS, "EN", "English") + collect(BN_ROOTS, "BN", "Bengali")
    with ThreadPoolExecutor(16) as ex: X_all = np.stack(list(ex.map(load_one, [r[0] for r in rows])))
    meta = pd.DataFrame(rows, columns=["path", "label", "lang"])
    np.savez_compressed(cache, X=X_all, meta=np.array(rows, dtype=object))
print(X_all.shape, meta.groupby("lang").size().to_dict(), "classes:", meta.label.nunique())

from sklearn.model_selection import train_test_split
h = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in X_all])
meta["hash"] = h
lab_per_hash = meta.groupby("hash").label.nunique()
conflict = set(lab_per_hash[lab_per_hash > 1].index)
dup_mask = meta.duplicated("hash", keep="first") | meta.hash.isin(conflict)
dup_report = {"total_images": int(len(meta)), "exact_duplicates_removed": int(dup_mask.sum()),
              "conflicting_label_hashes": int(len(conflict)),
              "removed_by_language": meta[dup_mask].groupby("lang").size().to_dict()}
print(dup_report)
keep = ~dup_mask.values
X_all, meta = X_all[keep], meta[keep].reset_index(drop=True)

classes = sorted(meta.label.unique()); cidx = {c: i for i, c in enumerate(classes)}
y_all = meta.label.map(cidx).values; NC = len(classes)
tr_f, va_f, te_f = CFG["SPLIT"]
idx = np.arange(len(y_all))
i_tr, i_tmp = train_test_split(idx, test_size=va_f + te_f, stratify=y_all, random_state=CFG["SEED"])
i_va, i_te = train_test_split(i_tmp, test_size=te_f / (va_f + te_f), stratify=y_all[i_tmp], random_state=CFG["SEED"])
print("originals  train/val/test:", len(i_tr), len(i_va), len(i_te))

def dhash(x):                                   # 64-bit difference hash on a 9x8 grey thumbnail
    g = np.asarray(Image.fromarray(x).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0]
DH = np.array([dhash(x) for x in X_all], dtype=np.uint64)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def min_dist(test_ids, train_ids):
    out = np.full(len(test_ids), 64, dtype=np.int16)
    for c in np.unique(y_all[test_ids]):
        t = test_ids[y_all[test_ids] == c]; r = DH[train_ids[y_all[train_ids] == c]]
        for k in range(0, len(t), 64):
            x = (DH[t[k:k+64]][:, None] ^ r[None, :]).view(np.uint8).reshape(len(t[k:k+64]), len(r), 8)
            out[np.isin(test_ids, t[k:k+64])] = POP[x].sum(-1).min(1)
    return out
md = min_dist(i_te, i_tr)
near = pd.DataFrame({"lang": meta.lang.values[i_te], "d": md})
near_report = {lang: {"test_images": int(len(g)), "identical_dhash_in_train_%": round(100 * (g.d == 0).mean(), 2),
                      "within_2_bits_%": round(100 * (g.d <= 2).mean(), 2), "within_5_bits_%": round(100 * (g.d <= 5).mean(), 2)}
               for lang, g in near.groupby("lang")}
print(json.dumps(near_report, indent=1))

from scipy.ndimage import affine_transform
rng = np.random.default_rng(CFG["SEED"])
def augment(img):                               # width/height shift, shear, zoom, nearest fill; no flip, no rotation
    s = CFG["IMG_STORE"]; sh = rng.uniform(-0.1, 0.1); z = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.1, 0.1, 2) * s
    A = np.array([[1 / z, sh / z], [0, 1 / z]]); c = np.array([s / 2, s / 2])
    off = c - A @ c - np.array([ty, tx])
    return np.stack([affine_transform(img[..., k], A, offset=off, mode="nearest", order=1) for k in range(3)], -1).astype(np.uint8)

tr_counts = pd.Series(y_all[i_tr]).value_counts()
target = int(tr_counts.max())
X_tr, y_tr, aug_flag = [X_all[i_tr]], [y_all[i_tr]], [np.zeros(len(i_tr), bool)]
for c, n in tr_counts.items():
    if n < target:
        src = i_tr[y_all[i_tr] == c]; pick = rng.choice(src, target - n, replace=True)
        X_tr.append(np.stack([augment(X_all[p]) for p in pick])); y_tr.append(np.full(len(pick), c)); aug_flag.append(np.ones(len(pick), bool))
X_tr, y_tr, aug_flag = np.concatenate(X_tr), np.concatenate(y_tr), np.concatenate(aug_flag)
X_va, y_va, X_te, y_te = X_all[i_va], y_all[i_va], X_all[i_te], y_all[i_te]
lang_te = meta.lang.values[i_te]; lang_va = meta.lang.values[i_va]
data_report = {"classes": NC, "originals_after_dedup": int(len(y_all)), "train_originals": int(len(i_tr)),
               "train_augmented_added": int(aug_flag.sum()), "train_total": int(len(y_tr)), "per_class_train": target,
               "val": int(len(y_va)), "test": int(len(y_te)),
               "test_by_language": pd.Series(lang_te).value_counts().to_dict()}
print(json.dumps(data_report, indent=1))

import tensorflow as tf, keras
from keras import layers
keras.utils.set_random_seed(CFG["SEED"])
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
if tf.config.list_physical_devices("GPU"): keras.mixed_precision.set_global_policy("mixed_float16")
_smi = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip().splitlines()
GPU_NAME = _smi[0] if _smi else "CPU"
print("TF", tf.__version__, "Keras", keras.__version__, "GPU:", tf.config.list_physical_devices("GPU"))
A = keras.applications
S = CFG["IMG_SIZE"]
PRE = {"MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input), "InceptionV3": (A.InceptionV3, A.inception_v3.preprocess_input),
       "VGG16": (A.VGG16, A.vgg16.preprocess_input), "Xception": (A.Xception, A.xception.preprocess_input),
       "DenseNet201": (A.DenseNet201, A.densenet.preprocess_input), "NASNetMobile": (A.NASNetMobile, A.nasnet.preprocess_input)}

def head(x, name):
    x = layers.Dropout(0.3)(x); x = layers.Dense(256, activation="relu", name="embed")(x)
    x = layers.Dropout(0.3)(x); out = layers.Dense(NC, activation="softmax", dtype="float32", name="prob")(x)
    return out

def build(name):
    inp = keras.Input((CFG["IMG_STORE"], CFG["IMG_STORE"], 3), name="image")
    x = layers.Resizing(S, S, name="resize")(inp)
    if name in PRE:
        fn, pp = PRE[name]
        x = layers.Lambda(pp, name="preprocess")(x)
        base = fn(include_top=False, weights="imagenet" if CFG["PRETRAINED"] else None, input_shape=(S, S, 3), pooling="avg")
        x = base(x)
    elif name == "AlexNet":                      # modified AlexNet: 5 conv + 3 FC, with batch normalization
        x = layers.Rescaling(1 / 255.)(x)
        for f, k, s_, pool in [(96, 11, 4, True), (256, 5, 1, True), (384, 3, 1, False), (384, 3, 1, False), (256, 3, 1, True)]:
            x = layers.Conv2D(f, k, strides=s_, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
            if pool: x = layers.MaxPooling2D(3, 2)(x)
        x = layers.Flatten()(x); x = layers.Dense(1024, activation="relu")(x)
    elif name == "BenSignNet":                   # compact BenSignNet-style CNN
        x = layers.Rescaling(1 / 255.)(x)
        for f in [32, 64, 128, 256]:
            for _ in range(2 if f < 256 else 1):
                x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
            x = layers.MaxPooling2D(2)(x)
        x = layers.GlobalAveragePooling2D()(x)
    m = keras.Model(inp, head(x, name), name=name)
    lr = CFG["LR_PRETRAINED"] if name in PRE else CFG["LR_SCRATCH"]
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
              metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5")])
    return m

MODELS = ["AlexNet", "BenSignNet", "MobileNetV2", "InceptionV3", "VGG16", "Xception", "DenseNet201", "NASNetMobile"]
def ds(X, y, shuffle):
    d = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: d = d.shuffle(20000, seed=CFG["SEED"], reshuffle_each_iteration=True)
    return d.batch(CFG["BATCH"]).map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
if args.models:
    MODELS = [m for m in MODELS if m in args.models.split(",")]

for name in MODELS:
    wpath, hpath = f"{OUT}/weights/{name}.weights.h5", f"{OUT}/histories/{name}.json"
    if os.path.exists(wpath) and os.path.exists(hpath):
        print(f"{name}: already trained, skipping"); continue
    keras.backend.clear_session(); keras.utils.set_random_seed(CFG["SEED"])
    m = build(name); t0 = time.time()
    cb = [keras.callbacks.EarlyStopping("val_loss", patience=CFG["PATIENCE"], restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-6)]
    hist = m.fit(ds(X_tr, y_tr, True), validation_data=ds(X_va, y_va, False), epochs=CFG["EPOCHS"], callbacks=cb, verbose=2)
    m.save_weights(wpath)
    H = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    H["train_seconds"] = time.time() - t0; H["params"] = int(m.count_params())
    H["device"] = GPU_NAME; H["tensorflow"] = tf.__version__; H["keras"] = keras.__version__
    json.dump(H, open(hpath, "w"))
    print(f"{name}: done in {H['train_seconds']/60:.1f} min")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, roc_auc_score, confusion_matrix
def metrics(y, P):
    yp = P.argmax(1); p, r, f, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    top5 = np.mean([y[i] in np.argsort(P[i])[-5:] for i in range(len(y))])
    out = {"accuracy": accuracy_score(y, yp), "precision": p, "recall": r, "f1": f, "top5": top5,
           "log_loss": log_loss(y, np.clip(P, 1e-7, 1), labels=range(NC))}
    try: out["auc_ovr"] = roc_auc_score(y, P, multi_class="ovr", average="macro")
    except Exception: out["auc_ovr"] = np.nan
    for lang in ["English", "Bengali"]:
        k = lang_te == lang if len(y) == len(lang_te) else lang_va == lang
        out[f"acc_{lang}"] = accuracy_score(y[k], yp[k])
    return out

rows = []
for name in MODELS:
    ppath = f"{OUT}/preds/{name}.npz"
    if not os.path.exists(ppath):
        keras.backend.clear_session(); m = build(name); m.load_weights(f"{OUT}/weights/{name}.weights.h5")
        fe = keras.Model(m.input, [m.get_layer("embed").output, m.output])
        Ev, Pv = fe.predict(ds(X_va, y_va, False), verbose=0); t0 = time.time()
        Et, Pt = fe.predict(ds(X_te, y_te, False), verbose=0); ms = 1000 * (time.time() - t0) / len(y_te)
        np.savez_compressed(ppath, Ev=Ev.astype(np.float32), Pv=Pv.astype(np.float32), Et=Et.astype(np.float32), Pt=Pt.astype(np.float32), ms=ms)
    z = np.load(ppath); H = json.load(open(f"{OUT}/histories/{name}.json"))
    r = metrics(y_te, z["Pt"]); r.update(model=name, params_M=H["params"] / 1e6, epochs=len(H["loss"]),
                                         train_min=H["train_seconds"] / 60, infer_ms_per_img=float(z["ms"]),
                                         best_val_acc=max(H["val_accuracy"]))
    rows.append(r)
single = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
single.round(4).to_csv(f"{OUT}/tables/single_models.csv"); print(single.round(4).to_string())

from scipy.stats import binomtest
PRED = {n: np.load(f"{OUT}/preds/{n}.npz") for n in MODELS}
top3_val = list(single.sort_values("best_val_acc", ascending=False).index[:3])
COMBOS = {"SignLink (AlexNet+MobileNetV2+Xception)": ["AlexNet", "MobileNetV2", "Xception"],
          "AlexNet+MobileNetV2": ["AlexNet", "MobileNetV2"], "AlexNet+Xception": ["AlexNet", "Xception"],
          "MobileNetV2+Xception": ["MobileNetV2", "Xception"],
          "AlexNet+DenseNet201+BenSignNet": ["AlexNet", "DenseNet201", "BenSignNet"],
          "DenseNet201+InceptionV3+VGG16": ["DenseNet201", "InceptionV3", "VGG16"],
          "Top-3 by validation (" + "+".join(top3_val) + ")": top3_val,
          "All eight backbones": MODELS}
COMBOS = {k: v for k, v in COMBOS.items() if all(n in PRED for n in v)}
strat = y_va if pd.Series(y_va).value_counts().min() >= 2 else None
i_fit, i_stop = train_test_split(np.arange(len(y_va)), test_size=0.5, stratify=strat, random_state=CFG["SEED"])

def fit_fusion(members, seed):
    keras.backend.clear_session(); keras.utils.set_random_seed(seed)
    Fv = np.concatenate([PRED[n]["Ev"] for n in members], 1); Ft = np.concatenate([PRED[n]["Et"] for n in members], 1)
    mu, sd = Fv[i_fit].mean(0), Fv[i_fit].std(0) + 1e-6
    inp = keras.Input((Fv.shape[1],)); x = layers.Dropout(0.3)(inp); out = layers.Dense(NC, activation="softmax", dtype="float32")(x)
    f = keras.Model(inp, out); f.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    f.fit((Fv[i_fit] - mu) / sd, y_va[i_fit], validation_data=((Fv[i_stop] - mu) / sd, y_va[i_stop]), epochs=200, batch_size=256, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)])
    return f.predict((Ft - mu) / sd, verbose=0)

def mcnemar(y, a, b):
    ca, cb_ = a == y, b == y; n01, n10 = int((ca & ~cb_).sum()), int((~ca & cb_).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)

fus_rows, case_rows, pair_rows, FUSED = [], [], [], {}
for cname, mem in COMBOS.items():
    runs = [fit_fusion(mem, CFG["SEED"] + s) for s in range(CFG["FUSION_SEEDS"])]
    accs = [accuracy_score(y_te, P.argmax(1)) for P in runs]
    Pf = runs[int(np.argsort(accs)[len(accs) // 2])]; FUSED[cname] = Pf           # median-accuracy run for analysis
    Pvote = np.mean([PRED[n]["Pt"] for n in mem], 0)
    preds = {n: PRED[n]["Pt"].argmax(1) for n in mem}
    best = max(mem, key=lambda n: accuracy_score(y_te, preds[n]))
    correct = np.stack([preds[n] == y_te for n in mem])
    dis = np.mean([np.mean(preds[a] != preds[b]) for i, a in enumerate(mem) for b in mem[i + 1:]]) if len(mem) > 1 else 0
    dfault = np.mean([np.mean(~correct[i] & ~correct[j]) for i in range(len(mem)) for j in range(i + 1, len(mem))]) if len(mem) > 1 else 0
    yf = Pf.argmax(1); fr = metrics(y_te, Pf)
    n01, n10, p = mcnemar(y_te, yf, preds[best])
    fus_rows.append({"combination": cname, "members": len(mem), "fusion_acc_mean": np.mean(accs), "fusion_acc_std": np.std(accs),
                     "fusion_f1": fr["f1"], "fusion_acc_English": fr["acc_English"], "fusion_acc_Bengali": fr["acc_Bengali"],
                     "soft_vote_acc": accuracy_score(y_te, Pvote.argmax(1)), "best_member": best,
                     "best_member_acc": accuracy_score(y_te, preds[best]), "oracle_acc": correct.any(0).mean(),
                     "gain_over_best_pp": 100 * (np.mean(accs) - accuracy_score(y_te, preds[best])),
                     "pairwise_disagreement": dis, "double_fault": dfault,
                     "rescued_vs_best": n01, "broken_vs_best": n10, "mcnemar_p": p})
    bw = preds[best] == y_te; fw = yf == y_te; allr = correct.all(0); anyr = correct.any(0)
    for lang in ["All", "English", "Bengali"]:
        k = np.ones(len(y_te), bool) if lang == "All" else lang_te == lang
        case_rows.append({"combination": cname, "subset": lang, "n": int(k.sum()),
                          "all_members_correct": int((allr & k).sum()),
                          "members_disagree_fusion_correct": int((~allr & anyr & fw & k).sum()),
                          "members_disagree_fusion_wrong": int((~allr & anyr & ~fw & k).sum()),
                          "all_members_wrong_fusion_correct": int((~anyr & fw & k).sum()),
                          "all_members_wrong_fusion_wrong": int((~anyr & ~fw & k).sum()),
                          "fusion_fixed_best_member": int((~bw & fw & k).sum()), "fusion_broke_best_member": int((bw & ~fw & k).sum())})
    print(f"{cname}: {np.mean(accs):.4f} ± {np.std(accs):.4f} (best member {best} {accuracy_score(y_te, preds[best]):.4f})")
fusion = pd.DataFrame(fus_rows).set_index("combination"); cases = pd.DataFrame(case_rows)
fusion.round(4).to_csv(f"{OUT}/tables/fusion_combinations.csv"); cases.to_csv(f"{OUT}/tables/fusion_cases.csv", index=False)
print(fusion.round(4).to_string()); print(cases.to_string())

SL = "SignLink (AlexNet+MobileNetV2+Xception)"
mem = COMBOS[SL]; best = fusion.loc[SL, "best_member"]
yb, yf = PRED[best]["Pt"].argmax(1), FUSED[SL].argmax(1)
_, _, fb, _ = precision_recall_fscore_support(y_te, yb, labels=range(NC), zero_division=0)
_, _, ff, sup = precision_recall_fscore_support(y_te, yf, labels=range(NC), zero_division=0)
pc = pd.DataFrame({"class": classes, "support": sup, f"f1_{best}": fb, "f1_SignLink": ff, "gain": ff - fb}).sort_values("gain", ascending=False)
pc.round(4).to_csv(f"{OUT}/tables/per_class_gain_signlink.csv", index=False)
conf = pd.DataFrame(confusion_matrix(y_te, yf, labels=range(NC)), index=classes, columns=classes)
conf.to_csv(f"{OUT}/tables/confusion_signlink.csv")
off = [(classes[i], classes[j], int(conf.values[i, j])) for i in range(NC) for j in range(NC) if i != j and conf.values[i, j] > 0]
pd.DataFrame(sorted(off, key=lambda t: -t[2])[:25], columns=["true", "predicted", "count"]).to_csv(f"{OUT}/tables/top_confusions_signlink.csv", index=False)
print(pc.head(12).round(3).to_string())

import matplotlib; import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "savefig.dpi": 300})
F = f"{OUT}/figures"
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for n in MODELS:
    H = json.load(open(f"{OUT}/histories/{n}.json"))
    ax[0].plot(range(1, len(H["val_accuracy"]) + 1), H["val_accuracy"], marker="o", ms=2.5, lw=1, label=n)
    ax[1].plot(range(1, len(H["val_loss"]) + 1), H["val_loss"], marker="o", ms=2.5, lw=1, label=n)
ax[0].set(xlabel="Epoch", ylabel="Validation accuracy", title="(a) Validation accuracy")
ax[1].set(xlabel="Epoch", ylabel="Validation loss", title="(b) Validation loss")
for a in ax: a.grid(lw=0.3)
ax[1].legend(fontsize=6.5, ncol=2); fig.tight_layout(); fig.savefig(f"{F}/fig_learning_curves.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 2.9)); fz = fusion.reset_index()
lbl = [c.split(" (")[0] if c.startswith("SignLink") else c.replace("Top-3 by validation", "Top-3 val") for c in fz.combination]
xx = np.arange(len(fz)); w = 0.27
ax.bar(xx - w, 100 * fz.best_member_acc, w, label="Best single member", color="#9bb7d4")
ax.bar(xx, 100 * fz.soft_vote_acc, w, label="Soft voting", color="#f2c48d")
ax.bar(xx + w, 100 * fz.fusion_acc_mean, w, yerr=100 * fz.fusion_acc_std, capsize=2, label="Feature fusion", color="#5a9e6f")
ax.set_xticks(xx); ax.set_xticklabels([l.replace("+", "+\n") if len(l) > 22 else l for l in lbl], fontsize=6.3)
lo = 100 * min(fz.best_member_acc.min(), fz.soft_vote_acc.min(), fz.fusion_acc_mean.min())
ax.set_ylim(max(0, lo - 3), 100); ax.set_ylabel("Test accuracy (%)"); ax.legend(fontsize=7, ncol=3, loc="lower right"); ax.grid(axis="y", lw=0.3)
fig.tight_layout(); fig.savefig(f"{F}/fig_fusion_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.45, 2.6)); multi = fz[fz.members > 1]
ax.scatter(100 * multi.pairwise_disagreement, multi.gain_over_best_pp, s=18, color="#2f6f9f")
for _, r in multi.iterrows():
    ax.annotate(str(fz.index[fz.combination == r.combination][0] + 1), (100 * r.pairwise_disagreement, r.gain_over_best_pp), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.axhline(0, color="k", lw=0.5); ax.set(xlabel="Mean pairwise disagreement of members (%)", ylabel="Gain over best member (pp)"); ax.grid(lw=0.3)
fig.tight_layout(); fig.savefig(f"{F}/fig_complementarity.png"); plt.close(fig)

top = pc[pc.gain != 0].head(12)
fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.barh(top["class"][::-1], 100 * top.gain[::-1], color=["#5a9e6f" if g > 0 else "#c96b6b" for g in top.gain[::-1]])
ax.set_xlabel(f"F1 gain of SignLink over {best} (pp)"); ax.tick_params(axis="y", labelsize=6.5); ax.grid(axis="x", lw=0.3)
fig.tight_layout(); fig.savefig(f"{F}/fig_perclass_gain.png"); plt.close(fig)

fixed = np.where((yb != y_te) & (yf == y_te))[0][:8]
if len(fixed):
    fig, ax = plt.subplots(1, len(fixed), figsize=(7.2, 1.45))
    for a, i in zip(np.atleast_1d(ax), fixed):
        a.imshow(X_te[i]); a.axis("off")
        a.set_title(f"true {classes[y_te[i]]}\n{best}: {classes[yb[i]]}", fontsize=5.8)
    fig.tight_layout(); fig.savefig(f"{F}/fig_fixed_examples.png"); plt.close(fig)

cm = conf.values / np.maximum(conf.values.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7.2, 6.6)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_yticks(range(NC)); ax.set_xticklabels(classes, rotation=90, fontsize=3.8); ax.set_yticklabels(classes, fontsize=3.8)
ax.set(xlabel="Predicted", ylabel="True"); fig.colorbar(im, fraction=0.03, pad=0.01); fig.tight_layout(); fig.savefig(f"{F}/fig_confusion_signlink.png"); plt.close(fig)
print(os.listdir(F))

env = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
       "gpu": [d.name for d in tf.config.list_physical_devices("GPU")], "config": CFG,
       "english_roots": EN_ROOTS, "bengali_roots": BN_ROOTS, "classes": classes}
env["gpu_name_this_run"] = GPU_NAME
env["trained_on"] = {n: json.load(open(f"{OUT}/histories/{n}.json")).get("device", "Google Colab GPU (first session)") for n in MODELS}
json.dump({"env": env, "duplicates": dup_report, "near_duplicates": near_report, "data": data_report}, open(f"{OUT}/tables/run_info.json", "w"), indent=1, default=str)
zpath = os.path.join(os.path.dirname(os.path.abspath(OUT)), "signlink_results.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["tables", "figures", "histories"]:
        for fpath in glob.glob(f"{OUT}/{sub}/*"): zf.write(fpath, f"{sub}/{os.path.basename(fpath)}")
print("Saved", zpath)

