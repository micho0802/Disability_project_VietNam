# =================== ASL Random BG + Combine + +20% Aug (parallel, pickling-safe) ===================
import os, math, uuid, random, shutil
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Optional: set start method early (Linux can keep fork; Windows needs spawn)
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

# ---------- Top-level helpers (must be picklable) ----------
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean, self.std = mean, std
    def __call__(self, img_tensor):
        noise = torch.randn_like(img_tensor) * self.std + self.mean
        return torch.clamp(img_tensor + noise, 0.0, 1.0)

def _init_opencv_worker():
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

def _grabcut_mask_np(img_bgr):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    pad = max(6, min(h, w) // 20)
    rect = (pad, pad, w - 2*pad, h - 2*pad)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 3, mode=cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except Exception:
        fg = np.ones((h, w), bool)
    return fg

def _random_bg_np(shape, pastel_bg: bool):
    if not pastel_bg:
        return np.random.randint(0, 256, shape, dtype=np.uint8)
    base = np.random.randint(100, 256, (1, 1, 3), dtype=np.uint8)
    noise = np.random.randint(-30, 30, shape, dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def worker_random_bg(args):
    """(fpath_str, label, out_dir_str, target_size, pastel_bg, labeled_flag) -> optional (dst_path, label)"""
    _init_opencv_worker()
    fpath_str, lbl, out_dir_str, target_size, pastel_bg, labeled = args
    f = Path(fpath_str)
    out_dir = Path(out_dir_str)

    img_bgr = cv2.imread(str(f))
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, tuple(target_size))
    mask = _grabcut_mask_np(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bg = _random_bg_np(img_rgb.shape, pastel_bg)
    comp = np.where(mask[..., None], img_rgb, bg).astype(np.uint8)

    if labeled:
        dst = out_dir / lbl / f.name
        if dst.exists():
            dst = out_dir / lbl / f"{f.stem}_rb_{uuid.uuid4().hex[:6]}{f.suffix}"
    else:
        dst = out_dir / f.name

    cv2.imwrite(str(dst), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    if random.random() < 0.02:
        return (str(dst), lbl or f.name)
    return None

def worker_copy_file(args):
    """(src_path_str, dst_path_str, label) -> optional (dst_path, label)"""
    src_path_str, dst_path_str, lbl = args
    src_p, dst_p = Path(src_path_str), Path(dst_path_str)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src_p, dst_p)
    except Exception:
        return None
    if random.random() < 0.02:
        return (str(dst_p), lbl)
    return None

def worker_augment_one(args):
    """(src_path_str, lbl, out_dir_str, target_size, labeled_flag) -> optional (dst_path, label)"""
    _init_opencv_worker()
    src_path_str, lbl, out_dir_str, target_size, labeled = args
    out_dir = Path(out_dir_str)
    try:
        from torchvision import transforms
        from PIL import Image
        import torch

        class _AddGaussianNoise:
            def __init__(self, mean=0.0, std=0.015): self.mean, self.std = mean, std
            def __call__(self, img_tensor):
                noise = torch.randn_like(img_tensor) * self.std + self.mean
                return torch.clamp(img_tensor + noise, 0.0, 1.0)

        aug = transforms.Compose([
            transforms.Resize(tuple(target_size), interpolation=Image.BILINEAR),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)], p=0.8),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05), shear=(-5, 5)),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            _AddGaussianNoise(mean=0.0, std=0.015),
            transforms.ToPILImage()
        ])

        base = Image.open(src_path_str).convert("RGB")
        aug_img = aug(base)
        new_name = f"{Path(src_path_str).stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
        dst = (out_dir / lbl / new_name) if labeled else (out_dir / new_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        aug_img.save(dst, quality=95)

        if random.random() < 0.02:
            return (str(dst), lbl or Path(src_path_str).stem)
    except Exception:
        return None
    return None

# ---------- The Pipeline (parallel) ----------
class ASLPipeline:
    """
    1) make_random_bg_dataset()   -> Add random RGB background (GrabCut for any background)
    2) combine_datasets()         -> Merge original + random-bg datasets
    3) augment_20_percent()       -> Add ~20% more using torchvision (ASL-safe)
    """

    def __init__(self, target_size=(224, 224), seed=42, pastel_bg=False, n_workers=None, backend="process"):
        self.target_size = target_size
        self.valid_exts = (".jpg", ".jpeg", ".png")
        self.pastel_bg = pastel_bg
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.backend = backend  # "process" (recommended) or "thread"
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Only used for quick, local (non-worker) ops if ever needed.
        self.augment = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.BILINEAR),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)], p=0.8),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05), shear=(-5, 5)),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.015),
            transforms.ToPILImage()
        ])

    # ---------------- Utility ----------------
    def _is_image(self, p: Path): return p.is_file() and p.suffix.lower() in self.valid_exts
    def _safe_mkdir(self, p: Path): p.mkdir(parents=True, exist_ok=True)

    def _show_grid(self, imgs_labels, title="Preview"):
        if not imgs_labels: 
            print(f"{title}: no samples.")
            return
        plt.figure(figsize=(10,10))
        for i, (img, lbl) in enumerate(imgs_labels[:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(img); plt.title(lbl, fontsize=14, weight="bold")
            plt.axis("off")
        plt.suptitle(title, fontsize=16); plt.tight_layout(); plt.show()

    # ---------------- Step 1: Random RGB background (parallel) ----------------
    def make_random_bg_dataset(self, in_dir, out_dir=None):
        src = Path(in_dir).resolve()
        out = Path(out_dir or f"{in_dir}_with_random_rgb_values").resolve()
        self._safe_mkdir(out)

        label_dirs = [d for d in src.iterdir() if d.is_dir()]
        labeled = bool(label_dirs)
        if labeled:
            for d in label_dirs:
                self._safe_mkdir(out / d.name)

        # Build jobs
        if labeled:
            jobs = [(str(f), d.name, str(out), self.target_size, self.pastel_bg, True)
                    for d in label_dirs for f in d.iterdir() if self._is_image(f)]
        else:
            jobs = [(str(f), "", str(out), self.target_size, self.pastel_bg, False)
                    for f in src.iterdir() if self._is_image(f)]

        previews = []
        with ProcessPoolExecutor(max_workers=self.n_workers, initializer=_init_opencv_worker) as ex:
            for r in tqdm(ex.map(worker_random_bg, jobs), total=len(jobs), desc=f"RandomBG {src.name}"):
                if r and len(previews) < 12:
                    previews.append(r)

        # Open preview images in main process
        sample_imgs = []
        for pth, lbl in previews[:9]:
            try:
                sample_imgs.append((Image.open(pth).convert("RGB"), lbl))
            except Exception:
                pass

        print(f"✅ Random background dataset saved: {out}")
        self._show_grid(sample_imgs, f"Random RGB Backgrounds: {src.name}")
        return str(out)

    # ---------------- Step 2: Combine (parallel copy) ----------------
    def combine_datasets(self, src_dirs, combined_dir):
        out = Path(combined_dir).resolve()
        self._safe_mkdir(out)

        # create all labels that exist anywhere
        label_set = set()
        for s in src_dirs:
            for d in os.listdir(s):
                if (Path(s) / d).is_dir():
                    label_set.add(d)
        for lbl in sorted(label_set):
            self._safe_mkdir(out / lbl)

        tasks = []
        for src in src_dirs:
            base = Path(src).name
            labeled_subdirs = [d for d in Path(src).iterdir() if d.is_dir()]
            if labeled_subdirs:
                for d in labeled_subdirs:
                    for f in d.iterdir():
                        if not self._is_image(f): continue
                        dst = out / d.name / f.name
                        if dst.exists():
                            dst = out / d.name / f"{base}_{f.name}"
                        tasks.append((str(f), str(dst), d.name))
            else:
                for f in [x for x in Path(src).iterdir() if self._is_image(x)]:
                    dst = out / f.name
                    if dst.exists():
                        dst = out / f"{base}_{f.name}"
                    tasks.append((str(f), str(dst), ""))

        previews = []
        with ThreadPoolExecutor(max_workers=min(64, self.n_workers * 4)) as ex:
            for r in tqdm(ex.map(worker_copy_file, tasks), total=len(tasks), desc="Combining"):
                if r and len(previews) < 12:
                    previews.append(r)

        sample_imgs = []
        for pth, lbl in previews[:9]:
            try:
                sample_imgs.append((Image.open(pth).convert("RGB"), lbl))
            except Exception:
                pass

        print(f"✅ Combined dataset saved: {out}")
        self._show_grid(sample_imgs, "Combined Dataset Preview")
        return str(out)

    # ---------------- Step 3: Augment +20% (parallel) ----------------
    def augment_20_percent(self, in_dir, out_dir=None, include_originals=True):
        src = Path(in_dir).resolve()
        out = Path(out_dir or f"{in_dir}_augmented").resolve()
        self._safe_mkdir(out)

        labeled_dirs = [d for d in src.iterdir() if d.is_dir()]
        labeled = bool(labeled_dirs)
        if labeled:
            for d in labeled_dirs:
                self._safe_mkdir(out / d.name)

        files = ([(d.name, f) for d in labeled_dirs for f in d.iterdir() if self._is_image(f)]
                 if labeled else [("", f) for f in src.iterdir() if self._is_image(f)])
        base_total = len(files)
        print(f"Found {base_total} images in {src.name}")

        # (optional) copy originals in parallel
        if include_originals:
            def _copy_orig(item):
                lbl, f = item
                try:
                    img = Image.open(f).convert("RGB").resize(self.target_size, Image.BILINEAR)
                    dst = (out / lbl / f.name) if labeled else (out / f.name)
                    if dst.exists():
                        dst = dst.with_name(f"{dst.stem}_orig_{uuid.uuid4().hex[:6]}{dst.suffix}")
                    img.save(dst, quality=95)
                except Exception:
                    pass
                return None

            with ThreadPoolExecutor(max_workers=min(64, self.n_workers * 4)) as ex:
                list(tqdm(ex.map(_copy_orig, files), total=len(files), desc="Copying originals"))

        to_add = int(math.ceil(base_total * 0.20))
        print(f"Augmenting +{to_add} images (~20%)...")

        # build augmentation plan
        plan = []
        if labeled:
            per = {d.name: [f for f in d.iterdir() if self._is_image(f)] for d in labeled_dirs}
            target = {k: max(0, int(math.ceil(len(v) * 0.20))) for k, v in per.items()}
            drift, keys, i = to_add - sum(target.values()), list(target.keys()), 0
            while drift != 0 and keys:
                target[keys[i % len(keys)]] += 1 if drift > 0 else -1
                drift += -1 if drift > 0 else 1
                i += 1
            for lbl, lst in per.items():
                if not lst or target[lbl] <= 0: continue
                for k in range(target[lbl]):
                    plan.append((str(lst[k % len(lst)]), lbl, str(out), self.target_size, True))
        else:
            lst = [str(f) for _, f in files]
            for k in range(to_add):
                plan.append((lst[k % len(lst)], "", str(out), self.target_size, False))

        previews = []
        with ProcessPoolExecutor(max_workers=self.n_workers, initializer=_init_opencv_worker) as ex:
            for r in tqdm(ex.map(worker_augment_one, plan), total=len(plan), desc="Augmenting"):
                if r and len(previews) < 12:
                    previews.append(r)

        sample_imgs = []
        for pth, lbl in previews[:9]:
            try:
                sample_imgs.append((Image.open(pth).convert("RGB"), lbl))
            except Exception:
                pass

        print(f"✅ Augmentation complete: {out} | Copied: {base_total if include_originals else 0}, Augmented: {len(plan)}")
        self._show_grid(sample_imgs, "Augmented (+20%) Preview")
        return str(out)

# =========================== EXAMPLE USAGE ===========================
if __name__ == "__main__":
    train_dir = "/home/mich02/Desktop/Disability_project_Vietnam/ai4li_VSL/asl_alphabet/asl_alphabet_train"

    pipe = ASLPipeline(
        target_size=(224, 224),
        pastel_bg=False,
        n_workers=max(1, os.cpu_count() - 2),  # keep a couple cores free
        backend="process"
    )

    train_rand = pipe.make_random_bg_dataset(train_dir)
    train_comb = pipe.combine_datasets([train_dir, train_rand], f"{train_dir}_combined")
    train_aug  = pipe.augment_20_percent(train_comb)
