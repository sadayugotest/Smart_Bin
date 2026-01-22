import matplotlib

matplotlib.use('Agg')  # กำหนด Matplotlib ให้ใช้ backend ที่ไม่ต้องการ GUI (เช่น ใช้ในการสร้างกราฟในพื้นหลัง)

from ultralytics import YOLO  # นำเข้าโมเดล YOLO จากไลบรารี Ultralytics
import tkinter as tk  # นำเข้าไลบรารี Tkinter สำหรับสร้าง GUI
from tkinter import ttk, filedialog, \
    messagebox  # นำเข้าโมดูลย่อยสำหรับ widgets เพิ่มเติม, กล่องโต้ตอบไฟล์, และกล่องข้อความ
import threading  # นำเข้า threading สำหรับการทำงานแบบ Multi-threading (เพื่อไม่ให้ GUI ค้างขณะเทรน)
import os  # นำเข้า os สำหรับการจัดการกับระบบปฏิบัติการ (เช่น การจัดการไฟล์และโฟลเดอร์)
import glob  # นำเข้า glob สำหรับค้นหาไฟล์/โฟลเดอร์ตามรูปแบบที่กำหนด
import yaml  # นำเข้า yaml สำหรับการอ่านและเขียนไฟล์ YAML (ใช้สำหรับไฟล์ data.yaml ของ YOLO)


class YOLOTrainerGUI:
    """
    คลาสสำหรับสร้าง GUI เพื่อเลือกโฟลเดอร์ข้อมูลและเทรนโมเดล YOLO
    """

    def __init__(self, root):
        self.root = root
        self.root.title("GUI สำหรับเทรน YOLO")  # ตั้งชื่อหน้าต่าง GUI
        self.root.geometry("600x450")  # กำหนดขนาดเริ่มต้นของหน้าต่าง (เพิ่มความสูงเล็กน้อยเพื่อการจัดวางที่ดีขึ้น)

        # ตั้งค่า Style สำหรับ Progressbar สีเขียว
        # Set Style for Green Progressbar
        style = ttk.Style(self.root)
        style.theme_use('default')  # ใช้ธีมเริ่มต้นของ Tkinter
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor='#d3d3d3',  # สีกรอบของ Progressbar
                        background='#00ff00')  # สีของ Progressbar

        self.selected_folder = None  # เก็บที่อยู่โฟลเดอร์ข้อมูลที่ผู้ใช้เลือก
        self.training_in_progress = False  # Flag เพื่อติดตามสถานะว่ากำลังเทรนอยู่หรือไม่
        self.epochs = 300  # จำนวน Epochs เริ่มต้นในการเทรน (สามารถปรับเปลี่ยนได้)
        self.batch = 32
        self.imgsz = 640
        self.class_project_name = None  # เก็บชื่อโปรเจกต์สำหรับบันทึกผลลัพธ์การเทรน

        self.create_widgets()  # เรียกเมธอดเพื่อสร้างส่วนประกอบ GUI

    def create_widgets(self):
        """
        สร้างและจัดวางส่วนประกอบ GUI ทั้งหมดภายในหน้าต่าง
        """
        self.top_frame = ttk.Frame(self.root, padding=20)  # เฟรมด้านบนสำหรับปุ่มเลือกโฟลเดอร์และ Label แสดงที่อยู่
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.select_button = ttk.Button(self.top_frame, text="เลือกโฟลเดอร์ข้อมูล",
                                        command=self.choose_folder)  # ปุ่มสำหรับเลือกโฟลเดอร์ข้อมูล
        self.select_button.pack(pady=10)

        self.folder_label = ttk.Label(self.top_frame,
                                      text="ที่อยู่ของโฟลเดอร์ข้อมูลจะปรากฏที่นี่")  # Label แสดงที่อยู่โฟลเดอร์ที่เลือก
        self.folder_label.pack(pady=10)

        # สร้าง Frame สำหรับจัดวางข้อมูลของจำนวน class, ชื่อ class และชื่อ Project ในรูปแบบ grid
        self.info_frame = ttk.Frame(self.top_frame)  # เฟรมสำหรับช่องกรอกข้อมูล
        self.info_frame.pack(pady=5, anchor="center")

        # แถวที่ 0 สำหรับ "จำนวน Class"
        self.class_count_label = ttk.Label(self.info_frame, text="จำนวน Class:")
        self.class_count_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)  # จัดวางด้วย grid (ชิดขวา)

        self.class_count_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกจำนวน Class
        self.class_count_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)  # จัดวางด้วย grid (ชิดซ้าย)

        # แถวที่ 1 สำหรับ "ชื่อ Class"
        self.class_name_label = ttk.Label(self.info_frame, text="ชื่อ Class (คั่นด้วยคอมม่า):")
        self.class_name_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)

        self.class_name_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกชื่อ Class
        self.class_name_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # แถวที่ 2 สำหรับ "ชื่อ Project"
        self.class_project_label = ttk.Label(self.info_frame, text="ชื่อ Project:")
        self.class_project_label.grid(row=2, column=0, sticky="e", padx=5, pady=2)

        self.class_project_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกชื่อ Project
        self.class_project_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # แถวที่ 3 สำหรับ "จำนวน Epochs"
        self.epochs_label = ttk.Label(self.info_frame, text="จำนวน Epochs:")
        self.epochs_label.grid(row=3, column=0, sticky="e", padx=5, pady=2)

        self.epochs_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกจำนวน Epochs
        self.epochs_entry.insert(0, str(self.epochs))  # ตั้งค่าค่าเริ่มต้นเป็น 300
        self.epochs_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # แถวที่ 4 สำหรับ "จำนวน ฺBatch"
        self.batch_label = ttk.Label(self.info_frame, text="จำนวน Batch:")
        self.batch_label.grid(row=4, column=0, sticky="e", padx=5, pady=2)

        self.batch_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกจำนวน Batch
        self.batch_entry.insert(0, str(self.batch))  # ตั้งค่าค่าเริ่มต้นเป็น 32
        self.batch_entry.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # แถวที่ 5 สำหรับ "จำนวน imgsz"
        self.imgsz_label = ttk.Label(self.info_frame, text="จำนวน Imgsz:")
        self.imgsz_label.grid(row=5, column=0, sticky="e", padx=5, pady=2)

        self.imgsz_entry = ttk.Entry(self.info_frame, width=30)  # ช่องกรอกจำนวน imgsz
        self.imgsz_entry.insert(0, str(self.imgsz))  # ตั้งค่าค่าเริ่มต้นเป็น 640
        self.imgsz_entry.grid(row=5, column=1, sticky="w", padx=5, pady=2)

        self.bottom_frame = ttk.Frame(self.root, padding=20)  # เฟรมด้านล่างสำหรับปุ่ม Train และแถบ Progress
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.train_button = ttk.Button(self.bottom_frame, text="เริ่มเทรน YOLO",
                                       command=self.train_action)  # ปุ่มสำหรับเริ่มการเทรน
        self.train_button.pack(pady=10)

        # Label แสดงข้อความ progress เป็นเปอร์เซ็นต์
        self.output_label = ttk.Label(self.bottom_frame, text="")
        self.output_label.pack(pady=10)

        # สร้าง Progressbar (สีเขียว) โดยกำหนด maximum เป็น 100
        self.progress_bar = ttk.Progressbar(self.bottom_frame, orient="horizontal",
                                            length=400, mode="determinate",
                                            style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10)
        self.progress_bar["value"] = 0  # ตั้งค่าเริ่มต้นเป็น 0
        self.progress_bar["maximum"] = 100  # ตั้งค่าสูงสุดเป็น 100 (เปอร์เซ็นต์)

    def choose_folder(self):
        """
        เปิดกล่องโต้ตอบให้ผู้ใช้เลือกโฟลเดอร์ข้อมูล
        """
        folder_path = filedialog.askdirectory(title="เลือกโฟลเดอร์ข้อมูล")  # เปิดกล่องโต้ตอบ
        if folder_path:  # ถ้าผู้ใช้เลือกโฟลเดอร์
            self.selected_folder = folder_path  # เก็บเส้นทางโฟลเดอร์
            self.folder_label.config(text=f"ที่อยู่ของโฟลเดอร์ข้อมูล: {folder_path}")  # อัปเดต Label
        else:  # ถ้าผู้ใช้ไม่ได้เลือกโฟลเดอร์
            self.folder_label.config(text="ยังไม่ได้เลือกโฟลเดอร์ข้อมูล")  # อัปเดต Label

    def train_action(self):
        """
        ดำเนินการเมื่อผู้ใช้กดปุ่ม 'เริ่มเทรน YOLO'
        จะตรวจสอบข้อมูล, ปิดการใช้งานปุ่ม, เริ่มเทรดการเทรน, และเริ่มติดตามความคืบหน้า
        """
        # ตรวจสอบว่าได้เลือกโฟลเดอร์ข้อมูลแล้วหรือไม่
        if not self.selected_folder:
            messagebox.showwarning("ยังไม่ได้เลือกโฟลเดอร์", "กรุณาเลือกโฟลเดอร์ข้อมูลก่อนเริ่มการเทรน")
            return

        # ดึงข้อมูลจากช่องกรอก
        class_count_str = self.class_count_entry.get().strip()
        class_names_str = self.class_name_entry.get().strip()
        project_name = self.class_project_entry.get().strip()
        epochs_str = self.epochs_entry.get().strip()
        batch_str = self.batch_entry.get().strip()
        imgsz_str = self.imgsz_entry.get().strip()

        # ตรวจสอบว่าช่องใส่ข้อมูลทั้งหมดได้กรอกครบถ้วนแล้วหรือไม่
        if not class_count_str or not class_names_str or not project_name or not epochs_str or not batch_str or not imgsz_str:
            messagebox.showwarning("ข้อมูลไม่ครบถ้วน", "กรุณากรอกข้อมูลในทุกช่องก่อนกดปุ่ม 'เริ่มเทรน YOLO'")
            return

        # ตรวจสอบความถูกต้องของจำนวน Class
        try:
            class_count = int(class_count_str)
            if class_count <= 0:
                messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Class ต้องเป็นจำนวนเต็มบวก")
                return
        except ValueError:
            messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Class ต้องเป็นตัวเลข")
            return

        # ตรวจสอบความถูกต้องของจำนวน Epochs
        try:
            epochs = int(epochs_str)
            if epochs <= 0:
                messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Epochs ต้องเป็นจำนวนเต็มบวก")
                return
            self.epochs = epochs  # เก็บค่า Epochs สำหรับการติดตามความคืบหน้า
        except ValueError:
            messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Epochs ต้องเป็นตัวเลข")
            return
        try:
            batch = int(batch_str)
            if batch <= 0:
                messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Batch ต้องเป็นจำนวนเต็มบวก")
                return
            self.batch = batch  # เก็บค่า Batch สำหรับการติดตามความคืบหน้า
        except ValueError:
            messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Batch ต้องเป็นตัวเลข")
            return
        try:
            imgsz = int(imgsz_str)
            if imgsz <= 0:
                messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Imgsz ต้องเป็นจำนวนเต็มบวก")
                return
            self.imgsz = imgsz  # เก็บค่า Imgsz สำหรับการติดตามความคืบหน้า
        except ValueError:
            messagebox.showerror("ข้อมูลไม่ถูกต้อง", "จำนวน Imgsz ต้องเป็นตัวเลข")
            return

        self.class_project_name = project_name  # เก็บชื่อโปรเจกต์สำหรับเส้นทางไฟล์ผลลัพธ์

        # แยกชื่อ class ที่คั่นด้วยคอมม่าเป็น List และลบช่องว่าง
        class_names_list = [name.strip() for name in class_names_str.split(',') if name.strip()]
        if not class_names_list:
            messagebox.showerror("ข้อมูลไม่ถูกต้อง", "กรุณาใส่ชื่อ Class อย่างน้อยหนึ่งชื่อ")
            return
        if len(class_names_list) != class_count:
            # แจ้งเตือนผู้ใช้หากจำนวน Class ที่ป้อนไม่ตรงกับจำนวนชื่อ Class ที่ระบุ
            messagebox.showwarning("คำเตือน",
                                   f"จำนวน Class ({class_count}) ไม่ตรงกับจำนวนชื่อ Class ที่ป้อน ({len(class_names_list)})")

        print("จำนวน Class:", class_count)
        print("ชื่อ Class:", class_names_list)
        print("ชื่อ Project:", self.class_project_name)
        print("จำนวน Epochs:", self.epochs)
        print("จำนวน Batch:", self.batch)
        print("จำนวน Imgsz:", self.imgsz)

        self.train_button.config(state='disabled')  # ปิดการใช้งานปุ่ม Train เพื่อป้องกันการกดซ้ำ
        self.output_label.config(text="กำลังเริ่มต้นการเทรน โปรดรอสักครู่...")  # อัปเดตข้อความสถานะ
        self.root.update_idletasks()  # บังคับอัปเดต GUI ทันทีเพื่อให้เห็นการเปลี่ยนแปลง
        self.progress_bar["value"] = 0  # รีเซ็ต Progressbar
        self.training_in_progress = True  # ตั้งค่า Flag ว่ากำลังเทรนอยู่
        self.poll_training_progress()  # เริ่ม polling เพื่อติดตามความคืบหน้า

        def run_training():
            """
            ฟังก์ชันนี้จะทำงานในเธรดแยกเพื่อดำเนินการเทรนโมเดล YOLO
            """
            yaml_path = 'data.yaml'  # กำหนดชื่อไฟล์ data.yaml ชั่วคราว
            try:
                # โหลดโมเดล YOLOv8s ที่ฝึกมาล่วงหน้า (สามารถเปลี่ยนเป็นโมเดลอื่นได้ตามต้องการ เช่น "yolov8n.pt")
                model = YOLO("yolo11n-obb.pt")

                # กำหนดค่าคอนฟิกสำหรับไฟล์ data.yaml ที่ YOLO ต้องการ
                data_config = {
                    'path': self.selected_folder,  # เส้นทางไปยังโฟลเดอร์ข้อมูลหลัก (train, val, test)
                    'nc': class_count,  # จำนวน Class
                    'names': class_names_list,  # ชื่อ Class ทั้งหมด
                    'train': 'train',  # ชื่อโฟลเดอร์สำหรับข้อมูล train
                    'val': 'val'  # ชื่อโฟลเดอร์สำหรับข้อมูล validation
                }

                # สร้างไฟล์ data.yaml ชั่วคราว
                with open(yaml_path, 'w') as f:
                    yaml.dump(data_config, f,
                              sort_keys=False)  # เขียนข้อมูลลงในไฟล์ YAML, sort_keys=False เพื่อให้ลำดับคงที่อ่านง่าย

                # เริ่มการเทรนโมเดล
                # พารามิเตอร์ต่างๆ สามารถปรับแต่งได้ตามความเหมาะสม
                result = model.train(
                    data=yaml_path,  # ไฟล์คอนฟิกข้อมูล
                    epochs=self.epochs,  # จำนวนรอบการเทรน
                    batch=self.batch,  # ขนาด Batch
                    imgsz=self.imgsz,  # ขนาดรูปภาพที่ใช้ในการเทรน
                    name=self.class_project_name,
                    hsv_h=0.05,
                    hsv_s=0.6,
                    hsv_v=0.5,
                    scale=0.8,
                    translate=0.2,
                    fliplr=0.5,
                    flipud=0.1,
                    mosaic=1.0,
                    mixup=0.5,
                    erasing=0.3,
                    lr0=0.0005,  # อัตราการเรียนรู้เริ่มต้น
                    lrf=0.0001,  # อัตราการเรียนรู้สุดท้าย
                    momentum=0.937,  # โมเมนตัมสำหรับ Optimizer
                    weight_decay=0.0005,  # การลดน้ำหนัก (L2 regularization)
                    augment=True,
                    patience=200,
                    verbose=False,
                    # ชื่อโปรเจกต์สำหรับบันทึกผลลัพธ์ (จะอยู่ใน runs/detect/{ชื่อโปรเจกต์})
                    exist_ok=True  # อนุญาตให้สร้างโฟลเดอร์ใหม่แม้ชื่อโปรเจกต์จะซ้ำ
                )
                # เมื่อเทรนเสร็จเรียบร้อย
                self.root.after(0, lambda: self.output_label.config(
                    text="เทรนเสร็จสิ้น! สามารถตรวจสอบผลลัพธ์ได้ที่โฟลเดอร์ runs/detect"))
                self.root.after(0, lambda: messagebox.showinfo("เสร็จสิ้น", "การเทรนโมเดลเสร็จสิ้นเรียบร้อยแล้ว!"))

            except Exception as e:
                # จัดการข้อผิดพลาดที่เกิดขึ้นระหว่างการเทรน
                self.root.after(0, lambda: self.output_label.config(text=f"เกิดข้อผิดพลาดในการเทรน: {e}"))
                self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเทรน: {e}"))
                print(f"Error during training: {e}")  # พิมพ์ข้อผิดพลาดลงใน Console สำหรับการ Debug
            finally:
                self.training_in_progress = False  # ตั้งค่า Flag ว่าการเทรนจบลงแล้ว
                self.root.after(0, lambda: self.train_button.config(state='normal'))  # เปิดการใช้งานปุ่ม Train อีกครั้ง
                # ลบไฟล์ data.yaml ชั่วคราวหลังจากใช้งานเสร็จ
                if os.path.exists(yaml_path):
                    os.remove(yaml_path)

        # เริ่มการทำงานของฟังก์ชัน run_training ในเธรดแยก เพื่อให้ GUI ยังคงตอบสนอง
        threading.Thread(target=run_training, daemon=True).start()

    def get_latest_results_file(self):
        """
        ค้นหาไฟล์ 'results.csv' ล่าสุดจากไดเรกทอรีโปรเจกต์ที่สร้างโดย YOLO
        (เนื่องจาก YOLO อาจสร้างโฟลเดอร์ MyProject, MyProject1, MyProject2...)
        """
        if not self.class_project_name:
            return None  # คืนค่า None ถ้าไม่มีชื่อโปรเจกต์

        # ค้นหาไดเรกทอรีทั้งหมดที่ขึ้นต้นด้วยชื่อโปรเจกต์ใน 'runs/detect'
        project_dirs = glob.glob(os.path.join("runs", "detect", f"{self.class_project_name}*"))
        latest_project_dir = None
        if project_dirs:
            # เลือกไดเรกทอรีที่ถูกแก้ไขล่าสุด (ซึ่งควรจะเป็นของรอบการเทรนปัจจุบัน)
            latest_project_dir = max(project_dirs, key=os.path.getmtime)

        if latest_project_dir:
            # สร้างเส้นทางเต็มไปยังไฟล์ results.csv
            results_file_path = os.path.join(latest_project_dir, "results.csv")
            if os.path.exists(results_file_path):
                return results_file_path
        return None  # คืนค่า None ถ้าหาไฟล์ไม่พบ

    def poll_training_progress(self):
        """
        ฟังก์ชันนี้จะถูกเรียกซ้ำๆ เพื่อติดตามและอัปเดตความคืบหน้าการเทรนจากไฟล์ results.csv
        """
        if self.training_in_progress:  # ตรวจสอบว่าการเทรนยังคงดำเนินอยู่หรือไม่
            try:
                results_file = self.get_latest_results_file()
                if results_file and os.path.exists(results_file):  # ตรวจสอบว่าไฟล์ results.csv มีอยู่หรือไม่
                    with open(results_file, "r") as f:
                        lines = f.readlines()
                        # ตรวจสอบว่ามีข้อมูล Epoch หลัง Header หรือไม่ (อย่างน้อย 2 บรรทัด)
                        if len(lines) > 1:
                            last_line = lines[-1]  # อ่านบรรทัดสุดท้าย (ข้อมูล Epoch ล่าสุด)
                            parts = last_line.split(",")  # แยกข้อมูลด้วยคอมม่า

                            # สมมุติว่า epoch อยู่ในคอลัมน์แรก (index 0)
                            if len(parts) > 0:
                                current_epoch_str = parts[0].strip()  # ดึงค่า Epoch
                                try:
                                    current_epoch = int(current_epoch_str)  # แปลงเป็นตัวเลข
                                    # จำกัด current_epoch ไม่ให้เกินจำนวน epochs ทั้งหมด เพื่อไม่ให้ Progressbar เกิน 100%
                                    current_epoch = min(current_epoch, self.epochs)
                                except ValueError:
                                    current_epoch = 0  # ตั้งเป็น 0 ถ้าไม่สามารถแปลงเป็นตัวเลขได้ (เช่น อ่าน Header)

                                if self.epochs > 0:  # ป้องกันการหารด้วยศูนย์
                                    percent = int((current_epoch / self.epochs) * 100)
                                else:
                                    percent = 0  # ถ้า epochs เป็น 0 ให้เปอร์เซ็นต์เป็น 0

                                # อัปเดตข้อความสถานะและ Progressbar
                                progress_text = f"กำลังเทรน... {percent}% (Epoch: {current_epoch}/{self.epochs})"
                                self.output_label.config(text=progress_text)
                                self.progress_bar["value"] = percent
                            else:
                                # กรณีบรรทัดไม่ถูกต้อง (เช่น เป็นบรรทัดว่างเปล่า)
                                self.output_label.config(text="กำลังรอข้อมูลความคืบหน้าการเทรน...")
                        else:
                            # กรณีมีแค่ Header หรือไฟล์เพิ่งถูกสร้างและยังไม่มีข้อมูล Epoch
                            self.output_label.config(text="กำลังเริ่มต้นการเทรนและสร้างไฟล์ผลลัพธ์...")
                            self.progress_bar["value"] = 0
                else:
                    # กรณีไฟล์ results.csv ยังไม่มี (การเทรนอาจเพิ่งเริ่มต้นหรือเกิดข้อผิดพลาดในการสร้างไฟล์)
                    self.output_label.config(text="กำลังรอไฟล์ results.csv...")
                    self.progress_bar["value"] = 0

            except Exception as e:
                # ดักจับข้อผิดพลาดทั่วไปที่อาจเกิดขึ้นขณะอ่าน/ประมวลผลไฟล์ results.csv
                self.output_label.config(text=f"เกิดข้อผิดพลาดในการตรวจสอบความคืบหน้า: {e}")
                print(f"Error polling progress: {e}")  # พิมพ์ข้อผิดพลาดใน Console

            # ตั้งเวลาให้เรียกฟังก์ชันนี้อีกครั้งใน 2 วินาที (เพื่อให้ Progressbar อัปเดตเรื่อยๆ)
            self.root.after(2000, self.poll_training_progress)


if __name__ == "__main__":
    # สร้างหน้าต่างหลักของ Tkinter
    root = tk.Tk()
    # สร้าง instance ของคลาส YOLOTrainerGUI
    app = YOLOTrainerGUI(root)
    # เริ่มต้นลูปหลักของ Tkinter เพื่อให้ GUI ทำงานและรอการตอบสนองจากผู้ใช้
    root.mainloop()
