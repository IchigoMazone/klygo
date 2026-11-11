# Thiết kế thực nghiệm Klygo

Ngày thiết kế: 05/11/2026. Đây là kế hoạch trước thực nghiệm chính. Các số 16/500/3 dưới đây là quy mô dự kiến, không phải kết quả đã thu được.

## 1. Hai loại bằng chứng

**Tính đúng của thư viện:** chạy bộ unittest hiện có; tách pass, fail, error, skip. Mock backend chỉ kiểm tra hợp đồng phần mềm. Chạy thêm pilot mô hình thật để kiểm tra nạp checkpoint, CUDA, chuyển đổi nhãn và bounding box.

**Chất lượng mô hình và hiệu năng pipeline:** dự đoán trên cùng tập ảnh và đánh giá bằng pycocotools. Không dùng bảng tổng hợp của workflow hiện tại để xếp hạng vì workflow phân bổ lớp khác nhau cho từng mô hình và chỉ chọn 1 ảnh dương/1 ảnh âm mỗi lớp.

## 2. Dữ liệu

- Nguồn dự kiến: COCO 2017 validation, nhãn instances_val2017.json từ kho COCO chính thức.
- Chọn ngẫu nhiên 500 image_id không hoàn lại từ danh sách ID sắp tăng, seed=20261105, trước khi chạy mô hình. Lưu danh sách ID, SHA-256 nhãn và ảnh, thuật toán chọn mẫu, phiên bản Python.
- Pilot: 16 ảnh đầu của manifest; kiểm tra khả năng chạy và ước lượng thời gian/bộ nhớ. Không dùng điểm pilot để chọn mô hình hoặc chỉnh ngưỡng.
- Chạy chính: cùng 500 ảnh cho tất cả mô hình; đánh giá đầy đủ 80 lớp COCO. Lớp không có ground truth trong tập con được xử lý theo COCOeval và báo cáo số lớp có hỗ trợ.
- Không loại ảnh khó hoặc thay ảnh khi mô hình dự đoán sai. Lỗi tải ảnh cần được sửa hoặc ghi nhận trước chạy chính, không âm thầm thay mẫu.
- Không gọi điểm trên tập con là điểm toàn bộ COCO val2017 hay điểm leaderboard.
- COCO15 xây từ COCO128 chỉ dùng minh họa/kiểm thử. Không dùng nó làm chứng cứ độc lập về khả năng tổng quát hóa.

## 3. Mô hình dự kiến

| Mô hình | Vai trò | Cách tích hợp |
| --- | --- | --- |
| YOLOv8n, yolov8n.pt | Baseline một giai đoạn | YOLODetect có sẵn của Klygo |
| Faster R-CNN MobileNetV3-Large FPN, COCO_V1 | Baseline hai giai đoạn | Subclass Detector qua backend Torchvision |
| YOLO-World v2 small, yolov8s-worldv2.pt | Mô hình có khả năng từ vựng mở | YOLOWorldDetect có sẵn của Klygo |

- Chốt danh sách dựa trên pilot tương thích và tài nguyên, không dựa trên điểm AP.
- D-FINE/Keras và Grounding DINO/Hugging Face được mô tả ở kiến trúc nếu mã có hỗ trợ; không tuyên bố đã thực nghiệm khi chưa chạy.
- Tất cả dùng trọng số công khai pretrained. Ghi nguồn, checksum, phiên bản và dữ liệu huấn luyện được công bố. Kết quả trên 80 lớp COCO của YOLO-World không chứng minh zero-shot trên lớp chưa thấy.
- Các model/backend là cấu hình hệ thống khác nhau. Chênh lệch điểm không tách được ảnh hưởng backend khỏi kiến trúc, trọng số và tiền xử lý.

## 4. Cấu hình đánh giá chất lượng

- batch=1, FP32; seed cố định. Không huấn luyện hoặc fine-tune trong phạm vi hiện tại.
- YOLO và YOLO-World: imgsz=640, conf=0.001, NMS IoU=0.7, max_det=100.
- Faster R-CNN: min_size=640, max_size=640, box_score_thresh=0.001, box_nms_thresh=0.7, box_detections_per_img=100. Đây là cấu hình đánh giá thống nhất giới hạn kích thước, không phải mặc định model zoo; ghi rõ trong bảng.
- Các phép resize/padding nội bộ có thể khác nhau. Ghi nguyên tắc resize từng backend; không tuyên bố tensor đầu vào giống nhau giữa kiến trúc.
- Với YOLO-World dùng vocabulary COCO của checkpoint, ghi đầy đủ vocabulary; không chọn prompt riêng theo ground truth từng ảnh.
- Giữ ngưỡng thấp cho AP, không lọc ở confidence=0.35 trước khi tính AP. Không thêm NMS lần hai ngoài backend cho phép đo chính.
- Chuyển xyxy sang xywh, ánh xạ tên lớp sang category_id COCO thật (không dùng trực tiếp chỉ số 0..79 hay chỉ số 91 lớp của Torchvision).
- COCOeval bbox: AP@[.50:.05:.95], AP50, AP75, maxDets=[1,10,100], area=all; báo cáo thêm theo kích thước khi có đủ mẫu.
- Precision/Recall/F1 micro tại confidence=0.25, IoU=0.50: dùng matching và ignore/crowd của COCOeval; số liệu này khác AP trung bình theo lớp. Không chọn ngưỡng tối ưu trên tập đánh giá.
- Lưu dự đoán thô; tính metric từ file dự đoán để có thể kiểm tra độc lập mà không chạy lại inference.

## 5. Đo thời gian

- Chuẩn bị ảnh RGB trong RAM trước timer. Báo cáo rõ không gồm tải mạng, đọc ảnh từ đĩa và thời gian nạp checkpoint.
- Timer Klygo bao gồm tiền xử lý của backend, suy luận, hậu xử lý nội bộ và đóng gói kết quả; warm-up 5 lần, CUDA synchronize trước/sau timer.
- Một mô hình trong GPU tại một thời điểm, batch=1, cùng thiết bị và precision.
- Phép đo ổn định dự kiến: 100 ảnh cố định trong manifest, 3 lượt; lưu từng thời gian thô, trung vị, p90, trung bình và biến thiên giữa lượt. Không suy ra các lần lặp là các mẫu dữ liệu độc lập.
- FPS được định nghĩa là 1000 / latency trung bình tính bằng ms cho xử lý tuần tự batch=1; không gọi là thông lượng tối đa.
- Chi phí Klygo: đo xen kẽ backend trực tiếp và Klygo trên cùng model instance hoặc trọng số, cùng lời gọi, cùng ảnh và tham số; kiểm tra tương đương kết quả trước khi diễn giải chênh lệch. Không so timer chỉ forward với timer toàn pipeline.

## 6. Thống kê và phân tích lỗi

- Báo cáo điểm của tập con cố định và quy mô mẫu. Nếu tính khoảng tin cậy, bootstrap theo ảnh, cùng mẫu bootstrap cho mọi mô hình; dùng COCOeval có xử lý đúng ảnh lặp hoặc bản sao ID và công bố phương pháp.
- Không sử dụng kiểm định ý nghĩa thống kê khi chưa có thiết kế phù hợp; không kết luận vượt trội phổ quát từ một máy/một tập con.
- Hình phân tích lỗi gồm ví dụ thành công và thất bại; ghi image_id, model, confidence và cách chọn ảnh. Không chỉ chọn ảnh đẹp.
- Đề xuất thêm một ablation confidence/NMS chỉ khi cần trả lời câu hỏi cụ thể; tách khỏi điểm benchmark chính, không dùng nó để tối ưu ngầm trên cùng tập đánh giá.

## 7. Tái lập và tiêu chí hoàn thành

Đầu ra lưu tại klygo/artifacts/report/: manifest, annotations được chọn, checkpoint, dự đoán, metric, latency thô, ảnh minh họa, log, environment và requirements khóa phiên bản. Mã thực nghiệm ở report/experiments/.

Pilot đạt khi: có CUDA thực thi thành công; checkpoint được nạp; xuất dự đoán đúng schema; metric chạy với nhãn thật; kết quả lặp lại ổn định ở mức hợp lý; không có lỗi tọa độ/ánh xạ lớp.

Chỉ viết kết luận định lượng sau khi đã chạy phép đo chính. Khi có lỗi kiểm thử, lưu log và phân loại; không thay số hoặc bỏ test để có báo cáo đẹp.

## 8. Nguồn phương pháp

- COCO: https://cocodataset.org/
- COCO evaluation: https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/detection-eval.htm
- COCOeval: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Cặp phiên bản PyTorch/Torchvision: https://pytorch.org/get-started/previous-versions/

Phiên bản PyTorch ban đầu chọn 2.8.0 + Torchvision 0.23.0, CUDA 12.6 theo bảng chính thức; khả năng chạy thực tế trên GPU cần pilot xác nhận.
