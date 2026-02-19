# 🗑️ DX Smart Bin — ถังขยะอัจฉริยะด้วย YOLO Object Detection

ระบบถังขยะอัจฉริยะที่ใช้ **YOLO (Ultralytics)** ตรวจจับชนิดขวดและสถานะฝาขวด เพื่อเปิดฝาถังขยะที่เหมาะสมโดยอัตโนมัติ ทำงานบนบอร์ด **Raspberry Pi** ผ่านการควบคุม GPIO

---

## 📋 สารบัญ

- [ภาพรวมระบบ](#overview)
- [ฟีเจอร์หลัก](#features)
- [ประเภทขยะที่รองรับ](#waste-types)
- [Hardware ที่ใช้](#hardware)
- [GPIO Pin Mapping](#gpio-pin-mapping)
- [การติดตั้ง](#installation)
- [โครงสร้างโปรเจกต์](#project-structure)
- [การใช้งาน](#usage)
- [Logic การตรวจจับ](#detection-logic)
- [GUI Interface](#gui-interface)

---

<a id="overview"></a>
## 🔍 ภาพรวมระบบ

```
[กล้อง USB/PiCamera]
        │
        ▼
[YOLO Model — ตรวจจับ Class]
        │
        ├── ตรวจพบ "Cap"     → แจ้งเตือน "เอาฝาออกก่อนครับ"
        ├── ตรวจพบ ขวดพลาสติก → เปิดฝาถัง 1 (สีน้ำเงิน)
        ├── ตรวจพบ ขวดทั่วไป  → เปิดฝาถัง 2 (สีเหลือง)
        ├── ตรวจพบ ขวดแก้ว/โลหะ → เปิดฝาถัง 3 (สีเขียว)
        └── ไม่พบวัตถุ        → แสดง ERROR
                │
                ▼
        [Raspberry Pi GPIO]
                │
                ▼
        [มอเตอร์ M1 / M2 / M3]
                │
                ▼
        [เปิด/ปิด ฝาถังขยะ]
```

---

<a id="features"></a>
## ✨ ฟีเจอร์หลัก

| ฟีเจอร์ | รายละเอียด |
|---|---|
| 🤖 AI Object Detection | ใช้ YOLO ตรวจจับชนิดขวดและฝาขวดแบบ Real-time |
| 🚪 Auto Lid Control | ควบคุมมอเตอร์เปิด-ปิดฝาถัง 3 ช่องผ่าน GPIO |
| 🎨 Color-Coded UI | แสดงสีบน GUI ตามประเภทขยะ (น้ำเงิน/เหลือง/เขียว/แดง) |
| 🔔 Sound Feedback | เล่นเสียงแจ้งเตือนเมื่อตรวจจับขยะสำเร็จหรือพบข้อผิดพลาด |
| ⚡ GPIO Trigger | ใช้ sensor/ปุ่ม (GPIO Pin 12) ทริกเกอร์การตรวจจับ |
| 🖥️ Tkinter GUI | หน้าจอแสดงผลขนาด 800×450 พร้อม Countdown Timer |
| 📷 Camera Thread | อ่านภาพจากกล้องแบบ Multi-thread ที่ ~30 FPS |

---

<a id="waste-types"></a>
## 🗂️ ประเภทขยะที่รองรับ

| Class ที่ YOLO ตรวจจับ | ความหมาย | ถังที่เปิด | สี UI |
|---|---|---|---|
| `Cap` | พบฝาขวด (ยังไม่ได้ถอด) | ❌ ไม่เปิด | 🔴 แดง |
| `Not_cap` | ไม่มีฝาขวด (พร้อมทิ้ง) | ✅ เปิดตามชนิด | — |
| `Mansome`, `Honey`, `Crystal` | ขวดพลาสติก (น้ำดื่ม/น้ำผึ้ง) | 🔵 M1 (ถัง 1) | 🔵 น้ำเงิน |
| `M100`, `Vitamilk` | ขวดนมถั่วเหลือง/เครื่องดื่ม | 🟢 M3 (ถัง 3) | 🟢 เขียว |
| `Milk2`, `Milk1` | กล่องนม / ขยะทั่วไป | 🟡 M2 (ถัง 2) | 🟡 เหลือง |
| `Coke` | ขวดแก้ว / โลหะ | 🟢 M3 (ถัง 3) | 🟢 เขียว |
| ไม่พบวัตถุ | ตรวจไม่พบขยะ | ❌ ไม่เปิด | 🔴 ERROR |

---

<a id="hardware"></a>
## 🔧 Hardware ที่ใช้

- **Raspberry Pi** (รองรับ RPi.GPIO)
- **กล้อง USB** หรือ **Raspberry Pi Camera** (1920×1080)
- **มอเตอร์ DC** 3 ตัว (ควบคุมฝาถังขยะ 3 ช่อง)
- **Sensor/Switch** สำหรับทริกเกอร์ (เชื่อมต่อ GPIO Pin 12)
- **จอแสดงผล** ขนาด 800×450 (HDMI หรือ DSI)

---

<a id="gpio-pin-mapping"></a>
## 🗺️ GPIO Pin Mapping

| GPIO (BCM) | ชื่อตัวแปร | หน้าที่ |
|---|---|---|
| `12` | `input_pin` | รับสัญญาณ Sensor/ปุ่มกด (Input) |
| `4` | `M1_up` | มอเตอร์ 1 — เปิดฝา (ถังพลาสติก 🔵) |
| `17` | `M1_down` | มอเตอร์ 1 — ปิดฝา |
| `27` | `M2_up` | มอเตอร์ 2 — เปิดฝา (ถังทั่วไป 🟡) |
| `22` | `M2_down` | มอเตอร์ 2 — ปิดฝา |
| `5` | `M3_up` | มอเตอร์ 3 — เปิดฝา (ถังแก้ว/โลหะ 🟢) |
| `6` | `M3_down` | มอเตอร์ 3 — ปิดฝา |

> ⚠️ ใช้โหมด **GPIO.BCM** ในการกำหนด Pin

---

<a id="installation"></a>
## 📦 การติดตั้ง

### 1. Clone / ดาวน์โหลดโปรเจกต์

```bash
git clone <https://github.com/sadayugotest/Small_Group.git>
cd "DX smart bin"
```

### 2. ติดตั้ง Dependencies

```bash
pip install ultralytics opencv-python Pillow RPi.GPIO
```

> หากใช้ Raspberry Pi OS อาจต้องติดตั้ง RPi.GPIO แยก:
> ```bash
> sudo apt-get install python3-rpi.gpio
> ```

### 3. วาง YOLO Model

วางไฟล์ model ไว้ที่โฟลเดอร์ `model/`:

```
DX smart bin/
└── model/
    └── All.pt      ← YOLO model หลัก
```

> ไฟล์ `cap.pt` และ `small.pt` ในโปรเจกต์คือ model สำรอง/เฉพาะทาง

### 4. ตรวจสอบการเชื่อมต่อกล้อง

```bash
ls /dev/video*
# ต้องพบ /dev/video0
```

---

<a id="project-structure"></a>
## 📁 โครงสร้างโปรเจกต์

```
DX smart bin/
├── main1.py          # โปรแกรมหลัก (M1/M2/M3 แยกอิสระ)
├── main2.py          # โปรแกรมสำรอง (ทดสอบ/variant)
├── cap.pt            # YOLO model สำหรับตรวจจับฝาขวด
├── small.pt          # YOLO model ขนาดเล็ก (lightweight)
├── model/
│   └── All.pt        # YOLO model หลัก (ตรวจจับทุก class)
├── PlaySound.py      # โมดูลเล่นเสียงแจ้งเตือน
└── README.md
```

---

<a id="usage"></a>
## ▶️ การใช้งาน

### รันโปรแกรม

```bash
python main1.py
```

> ต้องรันบน **Raspberry Pi** เท่านั้น เนื่องจากใช้ `RPi.GPIO`

### การทำงาน

1. **หน้าจอหลัก** — แสดง `"Input Waste"` และ `"วางขยะได้เลย"`
2. **วางขยะหน้ากล้อง** แล้วกดปุ่ม / Sensor (GPIO Pin 12)
3. ระบบจะ **ถ่ายภาพและวิเคราะห์** ด้วย YOLO ทันที
4. **ผลลัพธ์จะแสดง** พร้อมสีพื้นหลังตามประเภทขยะ
5. **มอเตอร์จะเปิดฝาถัง** ที่เหมาะสมเป็นเวลา ~0.4 วินาที
6. **Countdown 5 วินาที** จากนั้นระบบรีเซ็ตกลับหน้าจอหลัก

### Keyboard Shortcut (สำหรับทดสอบ)

| ปุ่ม | การทำงาน |
|---|---|
| `S` | ทริกเกอร์การตรวจจับด้วย keyboard (ไม่ต้องใช้ sensor) |

---

<a id="detection-logic"></a>
## 🧠 Logic การตรวจจับ

```
ตรวจจับ YOLO (confidence ≥ 0.70)
│
├── พบ "Cap"?
│   └── YES → แจ้ง "เอาฝาออกก่อนครับ" (ไม่เปิดถัง)
│
├── พบ Mansome/Honey/Crystal + Not_cap
│   └── → เปิด M1 (ถังพลาสติก 🔵)
│
├── พบ M100/Vitamilk + Not_cap
│   └── → เปิด M3 (ถังแก้ว/โลหะ 🟢)
│
├── พบ Milk1/Milk2
│   └── → เปิด M2 (ถังทั่วไป 🟡)
│
├── พบ Coke
│   └── → เปิด M3 (ถังแก้ว/โลหะ 🟢)
│
└── ไม่พบวัตถุใด → ERROR 🔴
```

> **หมายเหตุ:** ระบบตรวจจับฝาขวด (`Cap` / `Not_cap`) ก่อนเสมอ  
> หากยังมีฝาอยู่ จะไม่เปิดฝาถังและแจ้งเตือนให้ผู้ใช้ถอดฝาออกก่อน

---

<a id="gui-interface"></a>
## 🖥️ GUI Interface

| สถานะ | สีพื้นหลัง | ข้อความ |
|---|---|---|
| รอรับขยะ | ⬜ เทา | Input Waste / วางขยะได้เลย |
| ขวดพลาสติก | 🔵 น้ำเงิน | Plastic Waste / ขวดพลาสติก |
| ขยะทั่วไป | 🟡 เหลือง | General Waste / ขยะทั่วไป |
| ขวดแก้ว/โลหะ | 🟢 เขียว | Glass/Metal Waste / ขวดแก้ว/โลหะ |
| ยังมีฝาขวด | 🔴 แดง | เอาฝาออกก่อนครับ |
| ตรวจไม่พบ | 🔴 แดง | ERROR |

---

## 📝 หมายเหตุ

- โปรแกรมจะ cleanup GPIO อัตโนมัติเมื่อปิดผ่านปุ่ม **"ออกโปรแกรม"** หรือกด `Ctrl+C`
- `main1.py` และ `main2.py` มีความแตกต่างในฟังก์ชัน `reset_to_default()` — `main1.py` ควบคุมมอเตอร์แยกแต่ละตัว, `main2.py` ใช้ M1 ทุกกรณี
- ความละเอียดกล้องตั้งไว้ที่ **1920×1080** เพื่อความแม่นยำในการตรวจจับ
- ค่า Confidence Threshold ของ YOLO อยู่ที่ **0.70** (70%)

---

*DX Smart Bin — Intelligent Waste Sorting System powered by YOLO & Raspberry Pi*
