# Literature Synthesis Matrix (ร่าง v1)

ใช้รูปแบบของ RM02: แต่ละแถวคือ paper 1 ฉบับ ใช้ 6 คอลัมน์ มี 9 งาน ปี 2019–2026 แบ่ง 3 กลุ่มตามวิธีบอกตำแหน่งรหัส
ทุกฉบับเปิดอ่านต้นฉบับหรือหน้า abstract แล้ว ตัวเลขคัดมาจากที่ผู้เขียนรายงาน PDF อยู่ใน `papers/` (ไม่ขึ้น git)

## ตาราง

| # | Paper | Core Problem | Methods | Datasets | Evaluation Metrics | Key Contributions | Limitations / Research Gaps (เทียบกับโจทย์เรา) |
|---|---|---|---|---|---|---|---|
| 1 | Ershova et al., *YOLO-Barcode*, Computer Optics 48(4), 2024, doi:10.18287/2412-6179-CO-1424 | ตัวตรวจจับบาร์โค้ดเดิมผูกกับชนิดรหัส ช้า หรือแม่นน้อย | YOLO-based detector ตรวจจับและจำแนก 1D/2D หลายชนิด (รวม DataMatrix) | ZVZ-real, ZVZ-synth (สังเคราะห์ 27k), SE-barcode ใหม่ 1,813 ภาพ (สังเคราะห์ในฉากจริง, มีรหัสเล็กและหลายรหัสต่อภาพ); ZVZ-real มีสูงสุด 7 รหัสต่อภาพ | F1-score ของการตรวจจับ, เวลา | F1 98.6% บน ZVZ-real (เดิม 88.9%), เร็วกว่าเดิม 4 เท่า, ชุดข้อมูล SE-barcode | ชุดข้อมูลมีป้าย 4 มุม (quadrangle) แต่โมเดลทำนายแค่กรอบสี่เหลี่ยมไม่เอียง ("bounding boxes for objects without slope") **ไม่ให้ 4 มุม** และ **ไม่วัดว่าอ่านรหัสได้จริง** |
| 2 | Pandya et al., *Barcode and QR Code Object Detection: An Experimental Study on YOLOv8 Models*, IEEE INNOCOMP 2024, pp. 740–746, doi:10.1109/INNOCOMP63224.2024.00127 (มีฉบับ arXiv:2511.22937 ด้วย) | ดูว่าขนาดโมเดล YOLOv8 มีผลต่อการตรวจจับบาร์โค้ด/QR อย่างไร | เทียบ YOLOv8 n / s / m | ชุดข้อมูล Kaggle 31,078 ภาพ (train 28,696 / val 2,382), 1D + QR, ภาพ 416×416, ฝึก 10 epoch | precision, recall, F1, "accuracy" | n 88.95%, s 97.10%, m 94.10% | วัดแค่การตรวจจับ, ไม่วัดการอ่าน, รายละเอียดการทดลองน้อย, ไม่มี DataMatrix |
| 3 | Alsulami & Jhanjhi, *Deep learning framework for barcode localization and decoding using simulated UAV imagery*, Sci Rep 16:399, 2026, doi:10.1038/s41598-025-29720-w | ตรวจนับสินค้าคลังด้วยโดรน แสงน้อย เงา บดบัง | YOLOv8 หากรอบ + OpenCV ถอดรหัส | Roboflow 31,510 ภาพ (1D + QR; train 28,696 / val 2,382 / test 432 ซึ่ง train/val ตรงกับตัวเลขของ [2] ทุกตัว แต่ไม่แน่ใจว่าเป็นชุดเดียวกัน) + 300 ภาพเติมความเสื่อมจำลอง; ภาพตัวอย่าง (Fig. 6) เป็นรหัสใหญ่ 1 อันต่อภาพ ไม่ได้ระบุจำนวนรหัสต่อภาพ | mAP@0.5, decoding accuracy แยกตามสภาพ, เวลา, 6 รอบ (seed None, 11, 22, 33, 44, 55) | mAP@0.5 0.922 ± 0.0026 (6 รอบ), อ่านได้ 98.5% (แสงดี), 85.2% (แสงน้อย), 80.5% (เอียง), 78.3% (บดบัง), 72.1% (เบลอ) | ถอดรหัสจากกรอบสี่เหลี่ยม ไม่มี 4 มุม, **ไม่วิเคราะห์ตามความละเอียดภาพ**, ไม่มี DataMatrix, ไม่ได้เทียบโมเดลอื่นเอง (ยกตัวเลขจากงานอื่น), ไม่รายงานการอ่านผิด |
| 4 | Zharkov & Zagaynov, *Universal Barcode Detector via Semantic Segmentation*, ICDAR 2019, arXiv:1906.06281 | รหัสหลายชนิด ทั้งเอกสารและภาพถ่าย ต้องใช้โมเดลเดียว | semantic segmentation ไม่ถึง 33,000 weights แล้วใช้ minAreaRect | 17,000 ภาพ 21 ชนิด รวม DataMatrix, Muenster, ArTe-Lab | detection rate, เวลา | detection rate 0.989 บน ArTe-Lab, 3.8 ms บน GPU | ได้กรอบเอียง ไม่วัดการอ่าน |
| 5 | Quenum et al., *Fast, Accurate Barcode Detection in Ultra High-Resolution Images*, IEEE ICIP 2021, pp. 1019–1023, arXiv:2102.06868 | ภาพใหญ่เกิน 10k×10k ย่อแล้วรหัสหาย | RPN + Y-Net + ฟิตกรอบ | ภาพสังเคราะห์หลายชนิด รวม Data Matrix, Muenster, ArTe-Lab | mAP, AR, เวลา | mAP 0.937, YOLOv4 0.882, Mask R-CNN 0.466, 16 ms | จบที่กรอบ ไม่วัดการอ่าน |
| 6 | Yuan et al., *MU R-CNN*, Future Internet 11(9):197, 2019, doi:10.3390/fi11090197 | mask ของ Mask R-CNN ไม่ครบเพราะลาย QR ละเอียด | Mask R-CNN + UNet + dice loss | โรงงาน 14,000 ภาพ 18,409 QR และชุดอื่น | mask AP, box AP | mask AP 17.38 เทียบ 8.66 | QR เท่านั้น AP ยังต่ำ ไม่มีการทดลองอ่าน |
| 7 | Rybakova et al., *Improving Data Matrix mobile recognition via fast Hough transform and adaptive grid extractors*, Computer Optics 49(6):1182–1190, 2025, doi:10.18287/COJ1804 | อ่าน DataMatrix จากกล้องมือถือในสภาพควบคุมไม่ได้ ข้อมูล DataMatrix สาธารณะมีน้อย | หา finder pattern ด้วย fast Hough transform + แบ่งช่องแบบวนซ้ำ (ไม่ใช่ deep learning) | SE-DMTX-SYN-1000 (กึ่งสังเคราะห์ 1,000 รหัส, เปิดตัวที่ SPRA 2024) รหัสกินพื้นที่อย่างน้อย 25% ของภาพ | mean IoU ของตำแหน่ง, **recognition accuracy**, เวลาบน ARM | ผลลัพธ์เป็น 4 มุม (oriented quadrilateral) mIoU 0.889, อ่านได้ 0.903 (ZXing-cpp 0.858, libdmtx 0.665, ZXing 0.569), 35 ms | **รหัสเดียวต่อภาพ** และกินพื้นที่ ≥ 25% ของภาพ, สมมติว่ามีตัวหา ROI ให้ก่อนแล้ว ("assuming that the region of interest (ROI) has already been identified"), ไม่ใช่ deep learning, ไม่ได้ศึกษาหลายรหัสเล็กในภาพความละเอียดสูง |
| 8 | Liu et al., *A Data Matrix Code Recognition Method Based on L-Shaped Dashed Edge Localization Using Central Prior*, Sensors 24(13):4042, 2024, doi:10.3390/s24134042 | รหัสในโรงงานขอบนูนหรือขาด | YOLOv5s + ขอบ L เส้นประ + libdmtx | ทดสอบ 265 ภาพ 5 แบบ | recognition accuracy rate, เวลา | วิธีเต็ม 90.2% รหัสหมุน 85% | ไม่ระบุจำนวนรหัสต่อภาพ ไม่แยกตามความละเอียด |
| 9 | Corpuz et al., *QR Code Detection with Perspective Correction and Decoding in Real-World Conditions Using Deep Learning and Enhanced Image Processing*, VISAPP 2025, pp. 685–690 | ตัวอ่านเดิมพลาดเมื่อ QR เอียงหรือแสงแย่ | YOLOv5 + Canny 4 มุม + perspective + OpenCV | Roboflow 1,338 ภาพ QR | จำนวนที่อ่านได้, มุมสูงสุด, เวลา | อ่านได้ 459 รหัส ZBar 352 OpenCV 214 | ผู้เขียนระบุว่ายังต้องรองรับหลาย QR ในภาพเดียว |

**งานที่ใช้ประกอบแต่ไม่ได้อยู่ในตาราง** (เป็นโมเดลเปรียบเทียบหรือพื้นฐาน):
- Zarbo et al. 2009: ความสำคัญทางคลินิก
- He et al. 2017 (Mask R-CNN), Khanam & Hussain 2024 (YOLOv11), Sapkota et al. 2025 (YOLO26), Robinson et al. 2026 (RF-DETR): โมเดลที่เทียบ
- Tremblay et al. 2018: domain randomization

## สังเคราะห์ Common Gap (รูปแบบ X → Y → Z ของ RM02)

- **กลุ่มกรอบ [1] [2] [3]:** ตรวจจับแม่นและเร็ว แต่กรอบไม่แก้ความเอียง [3] เอียงแล้วอ่านได้เหลือ 80.5%
- **กลุ่ม mask [4] [5] [6]:** ได้รูปทรงจริงของรหัส แต่แปลงกลับเป็นกรอบ และไม่วัดการอ่าน
- **กลุ่มมุมหรือขอบ [7] [8] [9]:** จัดรหัสให้ตรงแล้วอ่านดีขึ้น แต่ [7] ทำ 1 รหัสต่อภาพ และ [9] ยังต้องทำให้รองรับหลายรหัส

**Common Gap:** ใน 9 งานนี้ ยังไม่มีงานที่ใช้ mask หา 4 มุมของรหัส DataMatrix ขนาดเล็กหลายอันในภาพเดียว แล้ววัดถึงขั้นอ่านได้และอ่านผิด แยกตาม px/module

**ประเภท Gap (RM02):**
- **Evaluation Gap:** วัดแค่การตรวจจับ ไม่วัดการอ่านและการอ่านผิดเมื่อมีหลายรหัส [1][2][4][5][6]
- **Dataset Gap:** ไม่มีชุดข้อมูลหลายรหัสเล็กต่อภาพที่มีป้าย 4 มุมแม่นยำ [1][7]
- **Application Gap:** ยังไม่มีงานในบริบทฉลากสไลด์พยาธิวิทยาที่ถ่ายทั้งแถว [1]–[9]

**ทำไมยังไม่มีใครทำ (ตอบข้อผิดพลาด "Lack of Depth" ของ RM02):**
- ป้าย 4 มุมของรหัสเล็กระดับพิกเซลติดด้วยมือแพงและไม่แม่น
- ภาพฉลากสไลด์จริงมีเลขตัวอย่างผู้ป่วย ซึ่งเป็นข้อมูลสุขภาพตาม PDPA เผยแพร่เป็นชุดข้อมูลไม่ได้
- งานบาร์โค้ดส่วนใหญ่มาจากโลจิสติกส์และค้าปลีก ซึ่งรหัสใหญ่และอ่านทีละอันได้ ปัญหาหลายรหัสเล็กในภาพเดียวจึงไม่ใช่โจทย์หลักของสาขานั้น

## ที่ยังไม่แน่ใจ / ต้องตรวจเพิ่ม
- **[2]:** ตีพิมพ์ใน IEEE INNOCOMP 2024 แล้ว ได้ฉบับจาก IEEE Xplore มาเทียบ ตัวเลขทุกตัวตรงกับฉบับ arXiv รายละเอียดการทดลองยังน้อย
- **SE-DMTX-SYN-1000 (SPRA 2024):** อ้างโดย [7] ยังไม่ได้เปิดต้นฉบับ ถ้าเปิดได้ควรใส่เป็นงานเกี่ยวข้อง เพราะเป็นชุดข้อมูล DataMatrix กึ่งสังเคราะห์ที่มีตำแหน่งแม่นยำ ใกล้กับแนวทางของเรา
- **Vezzali 2025 และ BarBeR:** benchmark กลางที่อาจแยกผลตาม PPE ยังไม่ได้อ่านฉบับเต็ม เพราะเว็บมีด่าน verification ต้องอ่านก่อนยืนยันว่า px/module ยังเป็น GAP
