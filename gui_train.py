import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from anomalib.data import Folder
from anomalib.models import Patchcore  # เปลี่ยนจาก Padim เป็น Patchcore
from anomalib.engine import Engine
from lightning.pytorch.callbacks import Callback

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class GuiProgressCallback(Callback):
    def __init__(self, max_epochs: int, on_update):
        super().__init__()
        self.max_epochs = max_epochs
        self.on_update = on_update  # function(percent:int, text:str)

    def on_train_start(self, trainer, pl_module):
        self.on_update(0, "เริ่มเทรน...")

    def on_train_epoch_end(self, trainer, pl_module):
        curr = int(trainer.current_epoch) + 1
        pct = int(100 * curr / max(1, self.max_epochs))
        self.on_update(pct, f"กำลังเทรน Epoch {curr}/{self.max_epochs}")

    def on_train_end(self, trainer, pl_module):
        self.on_update(100, "เทรนเสร็จสมบูรณ์")


class TrainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Anomalib PatchCore - Train GUI")  # อัปเดตชื่อหน้าต่าง
        self.geometry("1000x600")
        self.minsize(900, 520)

        # State
        self.dataset_root: Path | None = None
        self.results_root: Path = Path.cwd() / "results"
        self.max_epochs = ctk.IntVar(value=1)  # PatchCore มักใช้เพียง 1 epoch
        self.is_training = False

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # Top controls
        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Dataset Root:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.dataset_entry = ctk.CTkEntry(top, placeholder_text="เลือกโฟลเดอร์ Dataset ที่มีโครงสร้าง train/test ตาม MVTec", width=500)
        self.dataset_entry.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(top, text="เลือกโฟลเดอร์...", command=self.select_dataset).grid(row=0, column=2, padx=6, pady=6)

        ctk.CTkLabel(top, text="Epochs:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.epoch_entry = ctk.CTkEntry(top, width=100)
        self.epoch_entry.insert(0, str(self.max_epochs.get()))
        self.epoch_entry.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        # ช่อง Image Size ใน top controls
        ctk.CTkLabel(top, text="Image Size:").grid(row=3, column=0, padx=6, pady=6, sticky="w")
        self.imgsize_entry = ctk.CTkEntry(top, width=100)
        self.imgsize_entry.insert(0, "256")
        self.imgsize_entry.grid(row=3, column=1, padx=6, pady=6, sticky="w")

        # Sub-paths panel (defaults follow your train.py)
        paths = ctk.CTkFrame(self)
        paths.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        for i in range(2):
            paths.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(paths, text="โฟลเดอร์ย่อยภายใน Dataset (ค่าเริ่มต้นตาม train.py):").grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(0,8))
        ctk.CTkLabel(paths, text="normal_dir:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.normal_dir_entry = ctk.CTkEntry(paths)
        self.normal_dir_entry.insert(0, "train/good")
        self.normal_dir_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(paths, text="abnormal_dir:").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        self.abnormal_dir_entry = ctk.CTkEntry(paths)
        self.abnormal_dir_entry.insert(0, "test/ng")
        self.abnormal_dir_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(paths, text="normal_test_dir:").grid(row=3, column=0, sticky="e", padx=6, pady=6)
        self.normal_test_dir_entry = ctk.CTkEntry(paths)
        self.normal_test_dir_entry.insert(0, "test/good")
        self.normal_test_dir_entry.grid(row=3, column=1, sticky="ew", padx=6, pady=6)

        # Right panel: progress + logs + start button
        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(right, text="พร้อมสำหรับเทรน")
        self.progress_label.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        self.progress = ctk.CTkProgressBar(right)
        self.progress.set(0.0)
        self.progress.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 8))

        self.log_box = ctk.CTkTextbox(right, wrap="word")
        self.log_box.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)

        self.start_btn = ctk.CTkButton(right, text="เริ่มเทรน", command=self.on_start_train)
        self.start_btn.grid(row=3, column=0, padx=6, pady=6, sticky="ew")

        # Add a button to select results directory (optional, but helpful)
        ctk.CTkLabel(top, text="Results Dir:").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        self.results_entry = ctk.CTkEntry(top, width=500)
        self.results_entry.insert(0, str(self.results_root))
        self.results_entry.grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(top, text="เลือกโฟลเดอร์...", command=self.select_results_dir).grid(row=2, column=2, padx=6, pady=6)

        # (เพิ่ม) ส่วนปรับพารามิเตอร์ PatchCore
        params_frame = ctk.CTkFrame(top)
        params_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=(4, 2))
        params_frame.grid_columnconfigure(7, weight=1)

        ctk.CTkLabel(params_frame, text="Backbone").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.backbone_var = ctk.StringVar(value="resnet18")
        ctk.CTkOptionMenu(params_frame, values=["resnet18", "resnet50", "wide_resnet50_2", "efficientnet_b0"], variable=self.backbone_var, width=110)\
            .grid(row=0, column=1, padx=4, pady=4)

        ctk.CTkLabel(params_frame, text="Layers").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        self.layers_entry = ctk.CTkEntry(params_frame, width=120)
        self.layers_entry.insert(0, "layer2,layer3")
        self.layers_entry.grid(row=0, column=3, padx=4, pady=4)

        ctk.CTkLabel(params_frame, text="Coreset Ratio").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        self.coreset_entry = ctk.CTkEntry(params_frame, width=70)
        self.coreset_entry.insert(0, "1.0")  # ใช้ทั้งหมด เพราะรูปน้อย
        self.coreset_entry.grid(row=0, column=5, padx=4, pady=4)

        ctk.CTkLabel(params_frame, text="kNN").grid(row=0, column=6, padx=4, pady=4, sticky="w")
        self.knn_entry = ctk.CTkEntry(params_frame, width=60)
        self.knn_entry.insert(0, "5")
        self.knn_entry.grid(row=0, column=7, padx=4, pady=4, sticky="w")

        # แถวที่ 2: checkboxes
        self.pretrained_var = ctk.BooleanVar(value=True)
        self.grayscale_var = ctk.BooleanVar(value=False)
        self.blur_var = ctk.BooleanVar(value=True)
        self.normalize_var = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(params_frame, text="Pretrained", variable=self.pretrained_var).grid(row=1, column=0, padx=4, pady=2, sticky="w")
        ctk.CTkCheckBox(params_frame, text="Grayscale", variable=self.grayscale_var).grid(row=1, column=1, padx=4, pady=2, sticky="w")
        ctk.CTkCheckBox(params_frame, text="Blur", variable=self.blur_var).grid(row=1, column=2, padx=4, pady=2, sticky="w")
        ctk.CTkCheckBox(params_frame, text="Normalize", variable=self.normalize_var).grid(row=1, column=3, padx=4, pady=2, sticky="w")

        # (เพิ่ม) Offline Mode checkbox (ใต้ params_frame หลัง checkboxes อื่น)
        # NOTE: params_frame already created later; so move addition after its creation
        self.offline_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(params_frame, text="Offline Mode", variable=self.offline_var).grid(row=1, column=4, padx=4, pady=2, sticky="w")

    def select_dataset(self):
        path = filedialog.askdirectory(title="เลือกโฟลเดอร์ Dataset")
        if path:
            self.dataset_root = Path(path)
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, str(self.dataset_root))

    def select_results_dir(self):
        path = filedialog.askdirectory(title="เลือกโฟลเดอร์ Results")
        if path:
            self.results_root = Path(path)
            self.results_entry.delete(0, "end")
            self.results_entry.insert(0, str(self.results_root))

    def _append_log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    def _set_progress(self, percent: int, text: str):
        # Called from callback, must schedule on Tk main thread
        def _apply():
            self.progress.set(max(0.0, min(1.0, percent / 100.0)))
            self.progress_label.configure(text=text)
        self.after(0, _apply)

    def on_start_train(self):
        if self.is_training:
            messagebox.showinfo("กำลังเทรนอยู่", "ระบบกำลังเทรนอยู่ โปรดรอให้เสร็จก่อน")
            return

        # Read inputs
        dataset_path_text = self.dataset_entry.get().strip()
        if not dataset_path_text:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกโฟลเดอร์ Dataset")
            return

        self.dataset_root = Path(dataset_path_text)
        if not self.dataset_root.exists():
            messagebox.showerror("ข้อผิดพลาด", f"ไม่พบโฟลเดอร์: {self.dataset_root}")
            return

        try:
            ep = int(self.epoch_entry.get().strip())
            if ep <= 0:
                raise ValueError
            self.max_epochs = ctk.IntVar(value=ep)
        except Exception:
            messagebox.showerror("ข้อผิดพลาด", "จำนวน Epoch ต้องเป็นจำนวนเต็มบวก")
            return

        # Sub-paths
        normal_dir = self.normal_dir_entry.get().strip() or None
        abnormal_dir = self.abnormal_dir_entry.get().strip() or None
        normal_test_dir = self.normal_test_dir_entry.get().strip() or None

        # Validate minimal structure
        if normal_dir is None or normal_test_dir is None:
            messagebox.showerror("ข้อผิดพลาด", "ต้องระบุ normal_dir และ normal_test_dir")
            return

        # อ่านค่าความละเอียดรูป
        imgsize_text = self.imgsize_entry.get().strip()
        try:
            if "x" in imgsize_text:
                w, h = map(int, imgsize_text.lower().split("x"))
                img_size = (w, h)
            else:
                img_size = int(imgsize_text)
        except Exception:
            messagebox.showerror("ข้อผิดพลาด", "Image Size ต้องเป็นตัวเลข เช่น 256 หรือ 256x256")
            return

        # อ่านค่าพารามิเตอร์ PatchCore
        backbone = self.backbone_var.get().strip()
        layers_text = self.layers_entry.get().strip()
        layers = [l.strip() for l in layers_text.split(",") if l.strip()]
        try:
            coreset_ratio = float(self.coreset_entry.get().strip())
            if not (0 < coreset_ratio <= 1.0):
                raise ValueError
        except Exception:
            messagebox.showerror("ข้อผิดพลาด", "Coreset Ratio ต้องเป็นตัวเลขช่วง (0,1]")
            return
        try:
            num_neighbors = int(self.knn_entry.get().strip())
            if num_neighbors <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("ข้อผิดพลาด", "kNN ต้องเป็นจำนวนเต็มบวก")
            return
        pretrained = self.pretrained_var.get()
        grayscale = self.grayscale_var.get()
        blur = self.blur_var.get()
        normalize = self.normalize_var.get()
        offline = self.offline_var.get()

        if offline:
            # Force offline environment
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["NO_GCE_CHECK"] = "true"
            # Disable pretrained if offline
            if pretrained:
                self._append_log("Offline mode: ปิด pretrained (บังคับ)")
            pretrained = False

        # Disable button during training
        self.is_training = True
        self.start_btn.configure(state="disabled")
        self.progress.set(0.0)
        self.progress_label.configure(text="เตรียมเริ่มเทรน...")
        self.log_box.delete("1.0", "end")
        self._append_log(f"Dataset: {self.dataset_root}")
        self._append_log(f"normal_dir: {normal_dir}")
        self._append_log(f"abnormal_dir: {abnormal_dir}")
        self._append_log(f"normal_test_dir: {normal_test_dir}")
        self._append_log(f"epochs: {self.max_epochs.get()}")
        self._append_log(f"image_size: {img_size}")
        self._append_log(f"backbone: {backbone}")
        self._append_log(f"layers: {layers}")
        self._append_log(f"coreset_ratio: {coreset_ratio}")
        self._append_log(f"num_neighbors: {num_neighbors}")
        self._append_log(f"pretrained: {pretrained} grayscale:{grayscale} blur:{blur} normalize:{normalize} offline:{offline}")

        # Read results directory from entry
        results_path_text = self.results_entry.get().strip()
        if results_path_text:
            self.results_root = Path(results_path_text)
        try:
            self.results_root.mkdir(parents=True, exist_ok=True)
            # Try to write a temp file to check permissions
            testfile = self.results_root / ".__write_test__"
            with open(testfile, "w") as f:
                f.write("test")
            testfile.unlink()
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเขียนไปยังโฟลเดอร์ Results: {self.results_root}\n{e}")
            return

        # Run training in background thread
        t = threading.Thread(
            target=self._train_worker,
            args=(
                self.dataset_root, normal_dir, abnormal_dir, normal_test_dir,
                self.max_epochs.get(), self.results_root, img_size,
                backbone, layers, coreset_ratio, num_neighbors,
                pretrained, grayscale, blur, normalize, offline
            ),
            daemon=True,
        )
        t.start()

    def _train_worker(
        self,
        dataset_root: Path,
        normal_dir: str | None,
        abnormal_dir: str | None,
        normal_test_dir: str | None,
        epochs: int,
        results_root: Path,
        img_size,              # retained for logging only
        backbone: str,
        layers: list[str],
        coreset_ratio: float,
        num_neighbors: int,
        pretrained: bool,
        grayscale: bool,       # no longer applied
        blur: bool,            # no longer applied
        normalize: bool,       # no longer applied
        offline: bool,
    ):
        try:
            # Removed all transform / resize application. Original image sizes are used.
            datamodule = Folder(
                name="train_only_ok",
                root=dataset_root,
                normal_dir=normal_dir,
                abnormal_dir=abnormal_dir,
                normal_test_dir=normal_test_dir,
                mask_dir=None,
            )

            # Removed transform injection loop (not needed)
            # Try to reduce multiprocessing issues on Windows
            for attr in ("num_workers", "train_num_workers", "val_num_workers", "test_num_workers", "predict_num_workers"):
                if hasattr(datamodule, attr):
                    try:
                        setattr(datamodule, attr, 0)
                    except Exception:
                        pass

            # สร้างโมเดลด้วยพารามิเตอร์ที่ผู้ใช้เลือก
            # สร้างโมเดล (พร้อม fallback หากมี network error)
            try:
                model = Patchcore(
                    backbone=backbone,
                    layers=layers,
                    coreset_sampling_ratio=coreset_ratio,
                    num_neighbors=num_neighbors,
                    pre_trained=pretrained,
                )
            except Exception as e:
                if "hub" in str(e).lower() or "internet" in str(e).lower() or "download" in str(e).lower():
                    self._append_log("ไม่สามารถโหลด pretrained (อาจ offline) -> fallback เป็น pre_trained=False")
                    model = Patchcore(
                        backbone=backbone,
                        layers=layers,
                        coreset_sampling_ratio=coreset_ratio,
                        num_neighbors=num_neighbors,
                        pre_trained=False,
                    )
                else:
                    raise

            if offline:
                self._append_log("โหมด Offline: ปิดการเชื่อมต่อ Hub / การดาวน์โหลดทั้งหมด")

            # Engine with progress callback
            progress_cb = GuiProgressCallback(
                max_epochs=epochs,
                on_update=lambda pct, text: self._set_progress(pct, text),
            )
            engine = Engine(
                max_epochs=epochs,
                accelerator="auto",
                default_root_dir=results_root,
                callbacks=[progress_cb],
            )

            self._append_log("เริ่มเทรน...")
            engine.fit(datamodule=datamodule, model=model)
            self._append_log("เริ่มทดสอบ (test)...")
            engine.test(datamodule=datamodule, model=model)

            # Done
            def _done():
                self.is_training = False
                self.start_btn.configure(state="normal")
                self.progress.set(1.0)
                self.progress_label.configure(text="เทรนเสร็จสมบูรณ์")
                messagebox.showinfo("สำเร็จ", "การเทรนเสร็จสมบูรณ์")
            self.after(0, _done)

        except Exception as e:
            def _fail(e=e):
                self.is_training = False
                self.start_btn.configure(state="normal")
                self.progress_label.configure(text="เกิดข้อผิดพลาด")
                messagebox.showerror("ข้อผิดพลาด", str(e))
                self._append_log(f"Error: {e}")
                print(e)
            self.after(0, _fail)




if __name__ == "__main__":
    # แนะนำให้ปิดการใช้ CUDA ถ้าไม่มี GPU
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    app = TrainApp()
    app.mainloop()
