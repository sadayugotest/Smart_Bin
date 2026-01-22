import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from anomalib.models import Patchcore  # เปลี่ยนจาก Padim เป็น Patchcore
from anomalib.engine import Engine
from anomalib.data import PredictDataset
# --- robust collate: convert ImageItem -> batch (B,C,H,W) ---
try:
    # If available in your anomalib version
    from anomalib.data.dataclasses.torch.image import ImageBatch  # type: ignore
except Exception:
    ImageBatch = None

def _collate_image_items(batch):
    # batch: list of ImageItem or dict-like
    images = []
    paths = []
    for x in batch:
        img = getattr(x, "image", None) if hasattr(x, "image") else x.get("image", None)
        path = getattr(x, "image_path", None) if hasattr(x, "image_path") else x.get("image_path", None)
        if img is None or path is None:
            continue
        # ensure tensor shape (C,H,W)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            if img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0
        images.append(img)
        paths.append(path)
    images = torch.stack(images, dim=0)  # (B,C,H,W)
    if ImageBatch is not None:
        return ImageBatch(image=images, image_path=paths)
    return {"image": images, "image_path": paths}
# ------------------------------------------------------------
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
import warnings
import cv2  # เพิ่มสำหรับทำ colormap
import time  # เพิ่มสำหรับจับเวลา

# ลด FutureWarning จาก timm
warnings.filterwarnings("ignore", category=FutureWarning)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Anomalib PatchCore Detector")  # อัปเดตชื่อหน้าต่าง
        self.geometry("1600x900")  # ตั้งขนาดคงที่ 1600x900
        self.img_path = None
        self.img_panel = None
        self.result_label = None
        self.ckpt_path: Path | None = None  # เก็บ path ของ checkpoint ที่เลือก

        # Layout: ซ้าย 1 ส่วน, ขวา 3 ส่วน (รวม 4 ส่วน -> ขวา = 3/4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)  # เดิม 4 -> ปรับเป็น 3 เพื่อให้ขวา = 3/4
        self.grid_rowconfigure(0, weight=1)

        # Left panel
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        left_frame.grid_rowconfigure((0,1,2), weight=0)
        left_frame.grid_columnconfigure(0, weight=1)

        upload_btn = ctk.CTkButton(left_frame, text="อัพโหลดรูปภาพ", command=self.upload_image)
        upload_btn.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # เพิ่มปุ่มเลือกโมเดล และ label แสดง path
        select_ckpt_btn = ctk.CTkButton(left_frame, text="เลือกโมเดล (.ckpt)", command=self.select_model)
        select_ckpt_btn.grid(row=1, column=0, padx=20, pady=(0,10), sticky="ew")
        self.ckpt_label = ctk.CTkLabel(left_frame, text="ยังไม่เลือกโมเดล", font=("Arial", 14), anchor="w", justify="left", wraplength=320)
        self.ckpt_label.grid(row=2, column=0, padx=20, sticky="ew")

        # ย้ายปุ่มตรวจสอบและผลลัพธ์ลงไปข้างล่าง
        detect_btn = ctk.CTkButton(left_frame, text="ตรวจสอบ", command=self.detect_image)
        detect_btn.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        self.result_label = ctk.CTkLabel(left_frame, text="", font=("Arial", 20))
        self.result_label.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # เฟรมตัวเลือกจัดหมวด retrain: good / ng (ซ่อนเริ่มต้น)
        self.retrain_frame = ctk.CTkFrame(left_frame)
        self.retrain_frame.grid(row=5, column=0, padx=20, pady=(0,20), sticky="ew")
        self.retrain_frame.grid_columnconfigure((0,1), weight=1)
        ctk.CTkLabel(self.retrain_frame, text="ผลใกล้เคียง 90–99% เลือกจัดหมวด:").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(6,6))
        ctk.CTkButton(self.retrain_frame, text="GOOD", fg_color="#2AA876", command=lambda: self._save_for_retrain("good")).grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(self.retrain_frame, text="NG", fg_color="#D7263D", command=lambda: self._save_for_retrain("ng")).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        self.retrain_frame.grid_remove()

        # Right panel (2 columns: master | anomaly)
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        self.master_label = ctk.CTkLabel(self.right_frame, text="Master")
        self.master_label.grid(row=0, column=0, sticky="nw")
        self.anom_label = ctk.CTkLabel(self.right_frame, text="Anomaly Map")
        self.anom_label.grid(row=0, column=1, sticky="nw")

        # อัพเดทภาพเมื่อ resize window
        self.bind("<Configure>", self.on_resize)

        # เก็บภาพต้นฉบับ/แผนที่ความผิดปกติ (ไม่ scale)
        self.master_pil: Image.Image | None = None
        self.anomaly_pil: Image.Image | None = None

    def on_resize(self, event):
        self._render_prepared_images()

    def _fit_size(self, w, h, maxw, maxh):
        # แสดงขนาดจริง แต่ถ้าใหญ่กว่า area ให้ย่อ (ไม่ขยายเกินของเดิม)
        scale = min(1.0, min(maxw / max(1, w), maxh / max(1, h)))
        return int(w * scale), int(h * scale)

    def _render_prepared_images(self):
        # คำนวณพื้นที่ของแต่ละช่องใน panel ขวา
        right_w = self.right_frame.winfo_width() or 1200  # 1600 * 3/4
        right_h = self.right_frame.winfo_height() or 900
        slot_w = max(10, right_w // 2 - 8)  # หาร 2 คอลัมน์
        slot_h = max(10, right_h - 8)

        if self.master_pil is not None:
            mw, mh = self.master_pil.size
            tw, th = self._fit_size(mw, mh, slot_w, slot_h)
            m_ctk = ctk.CTkImage(light_image=self.master_pil, size=(tw, th))
            self.master_label.configure(image=m_ctk, text="")
            self.master_label.image = m_ctk

        if self.anomaly_pil is not None:
            aw, ah = self.anomaly_pil.size
            tw, th = self._fit_size(aw, ah, slot_w, slot_h)
            a_ctk = ctk.CTkImage(light_image=self.anomaly_pil, size=(tw, th))
            self.anom_label.configure(image=a_ctk, text="")
            self.anom_label.image = a_ctk

    def show_image(self, file_path):
        # เก็บภาพจริง (ไม่ scale) แล้วให้ _render_prepared_images เป็นคนจัดการ
        img = Image.open(file_path).convert("RGB")
        self.master_pil = img
        self._render_prepared_images()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.img_path = file_path
            self.show_image(file_path)
            self.result_label.configure(text="")
            self._toggle_retrain_buttons(False)

    # ย่อ path สำหรับแสดงผลใน label
    def _shorten_path(self, p: Path) -> str:
        try:
            return f".../{p.parent.name}/{p.name}"
        except Exception:
            return str(p)

    # เลือกไฟล์โมเดล .ckpt
    def select_model(self):
        file_path = filedialog.askopenfilename(
            title="เลือกไฟล์โมเดล (.ckpt)",
            filetypes=[("Checkpoint", "*.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.ckpt_path = Path(file_path)
            self.ckpt_label.configure(text=self._shorten_path(self.ckpt_path))

    # ค้นหา checkpoint อัตโนมัติใต้ results/Patchcore
    def _auto_find_ckpt(self) -> Path | None:
        base = Path.cwd() / "results" / "Patchcore"  # เปลี่ยนจาก Padim เป็น Patchcore
        patterns = ["**/weights/**/*.ckpt", "**/weights/*.ckpt", "**/model.ckpt", "**/*.ckpt"]
        candidates = []
        for pat in patterns:
            candidates.extend(base.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _make_anomaly_pil(self, anomaly_map) -> Image.Image:
        # anomaly_map: torch.Tensor (H,W) หรือ (1,H,W)
        if isinstance(anomaly_map, torch.Tensor):
            amap = anomaly_map.detach().cpu().squeeze().float().numpy()
        else:  # numpy
            amap = np.array(anomaly_map, dtype=np.float32)
        if amap.size == 0:
            # fall back empty
            return Image.new("RGB", (320, 240), color=(30, 30, 30))
        # normalize 0-255
        a_min, a_max = float(np.min(amap)), float(np.max(amap))
        if a_max > a_min:
            amap = (amap - a_min) / (a_max - a_min)
        else:
            amap = np.zeros_like(amap, dtype=np.float32)
        amap = (amap * 255).clip(0, 255).astype(np.uint8)
        # colorize with JET
        heat_bgr = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heat_rgb)

    def _toggle_retrain_buttons(self, show: bool):
        if show:
            self.retrain_frame.grid()
        else:
            self.retrain_frame.grid_remove()

    def _save_for_retrain(self, label: str):
        # label in {"good", "ng"}
        if not self.img_path:
            return
        target_dir = Path.cwd() / "Retrain" / label
        target_dir.mkdir(parents=True, exist_ok=True)
        src = Path(self.img_path)
        # ป้องกันชื่อซ้ำ: เพิ่ม timestamp หน้าไฟล์
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = target_dir / f"{ts}_{src.name}"
        try:
            shutil.copy2(src, dst)
            self.result_label.configure(text=f"บันทึกรูปไปที่ {self._shorten_path(dst)}")
        except Exception as e:
            self.result_label.configure(text=f"บันทึกล้มเหลว: {e}")
        finally:
            # ซ่อนปุ่มหลังเลือกแล้ว
            self._toggle_retrain_buttons(False)

    def detect_image(self):
        if not self.img_path:
            self.result_label.configure(text="กรุณาอัพโหลดรูปก่อน")
            return

        try:
            # โหลดโมเดล/เช็คพอยต์: ใช้ Patchcore ให้ตรงกับที่เทรน
            model = Patchcore(
                backbone="resnet18",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
                num_neighbors=9,
                pre_trained=False,
            )
            engine = Engine(max_epochs=1, accelerator="auto", default_root_dir=Path.cwd() / "results")

            # หา ckpt: ใช้ที่ผู้ใช้เลือกก่อน ถ้าไม่มีให้หาอัตโนมัติ
            ckpt_path = self.ckpt_path or self._auto_find_ckpt()
            if not ckpt_path or not ckpt_path.exists():
                self.result_label.configure(text="ไม่พบโมเดล .ckpt กรุณาเลือกโมเดลก่อน")
                return

            # PredictDataset + DataLoader พร้อม collate ที่ให้รูปแบบ 4D (B,C,H,W)
            inference_dataset = PredictDataset(path=self.img_path)
            inference_dataloader = DataLoader(
                dataset=inference_dataset,
                batch_size=1,
                num_workers=0,           # สำคัญบน Windows
                collate_fn=_collate_image_items
            )

            # วัดเวลาเฉพาะช่วง predict
            t0 = time.perf_counter()
            predictions = engine.predict(
                model=model,
                dataloaders=inference_dataloader,
                ckpt_path=ckpt_path,
                return_predictions=True
            )[0]
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            # รองรับทั้งแบบ object และ dict
            if hasattr(predictions, "pred_score"):
                score = round(100 * (1 - predictions.pred_score[0].item()), 2)
            else:
                score = round(100 * (1 - predictions["pred_scores"][0].item()), 2)

            # ดึง anomaly map
            amap = None
            if hasattr(predictions, "anomaly_maps"):
                amap = predictions.anomaly_maps[0]
            elif isinstance(predictions, dict) and "anomaly_maps" in predictions:
                amap = predictions["anomaly_maps"][0]
            elif hasattr(predictions, "anomaly_map"):
                amap = predictions.anomaly_map[0]
            elif isinstance(predictions, dict) and "anomaly_map" in predictions:
                amap = predictions["anomaly_map"][0]

            if amap is not None:
                self.anomaly_pil = self._make_anomaly_pil(amap)
            else:
                # ไม่มี anomaly map ให้สร้างภาพแจ้งเตือน
                self.anomaly_pil = Image.new("RGB", (320, 240), color=(50, 50, 50))

            self.result_label.configure(
                text=f"ผลการตรวจสอบ: {score}%\nเวลา: {elapsed_ms} ms\nโมเดล: {self._shorten_path(ckpt_path)}"
            )
            self._render_prepared_images()

            # แสดงปุ่ม retrain หากผลอยู่ช่วง 90–99%
            self._toggle_retrain_buttons(90.0 <= score < 100.0)

            # อัพเดต label path หากมาจาก auto-find
            if self.ckpt_path is None:
                self.ckpt_label.configure(text=self._shorten_path(ckpt_path))
        except Exception as e:
            self.result_label.configure(text=f"เกิดข้อผิดพลาด: {e}")
            self._toggle_retrain_buttons(False)

if __name__ == "__main__":
    app = App()
    app.mainloop()
