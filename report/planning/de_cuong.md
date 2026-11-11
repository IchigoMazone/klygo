# Đề cương báo cáo Klygo

Ngày lập: 05/11/2026. Trạng thái: đề cương và thiết kế thực nghiệm; chưa phải báo cáo hoàn chỉnh.

## Thông tin đã thống nhất

- Đại học Phenikaa, Khoa Khoa học Máy tính.
- Học phần: Nhập môn Trí tuệ nhân tạo.
- Giảng viên: TS. Trần Đình Tân.
- Đề tài: **Thiết kế và xây dựng thư viện Klygo hỗ trợ xử lý dữ liệu và đánh giá các mô hình phát hiện đối tượng**.
- Người dùng xác nhận Klygo do nhóm phát triển. Nhóm gồm ba thành viên; tên, MSSV và phân công sẽ bổ sung sau.
- Nội dung tiếng Việt, giữ thuật ngữ tiếng Anh khi cần thiết. Kế thừa hình thức LaTeX từ report/main.tex, không kế thừa kết quả hay cam đoan của đề tài cũ.
- Mục tiêu 40–50 trang nội dung chính; phương án phân bổ dưới đây có tổng 46 trang. Không kéo dài bằng cách liệt kê API.
- Chạy mô hình pretrained và kiểm thử thực tế. Không mô tả đây là huấn luyện từ đầu hoặc một kiến trúc mạng mới của nhóm.
- Hạn nộp chưa được cung cấp. Chưa tự điền ngày nộp.

## Câu hỏi nghiên cứu

1. Có thể tổ chức pipeline phát hiện đối tượng qua nhiều backend bằng giao diện và cấu trúc đầu ra thống nhất như thế nào?
2. Klygo bảo toàn tọa độ, cấu hình và vòng đời mô hình như thế nào khi tiền xử lý, suy luận theo lô hoặc streaming?
3. Trên cùng dữ liệu đánh giá, các cấu hình mô hình khảo sát có sự đánh đổi nào giữa chất lượng và thời gian xử lý?
4. Lớp tích hợp Klygo bổ sung bao nhiêu chi phí thời gian so với lời gọi backend tương đương, trong phạm vi phép đo đã kiểm soát?

## Phần đầu

Bìa chính, bìa phụ, lời cảm ơn, lời cam đoan phù hợp với nguồn gốc mã và checkpoint; bảng phân công cần rà soát theo công việc thực tế; tóm tắt tiếng Việt và Abstract tiếng Anh (người dùng đã xác nhận giữ cả hai); mục lục; danh mục hình, bảng và từ viết tắt.

## Chương 1. Giới thiệu (4 trang)

1.1. Bối cảnh ứng dụng phát hiện đối tượng và vấn đề tích hợp nhiều thư viện.

1.2. Bài toán: đầu vào ảnh/video và cấu hình; đầu ra bounding box, nhãn, độ tin cậy và metadata.

1.3. Mục tiêu, câu hỏi nghiên cứu, tiêu chí đánh giá.

1.4. Phạm vi: thư viện Python; suy luận pretrained; đánh giá hữu hạn trên một tập dữ liệu và một máy.

1.5. Đóng góp của nhóm và phần kế thừa: mã Klygo, thiết kế thí nghiệm, mô hình và dữ liệu bên ngoài.

## Chương 2. Cơ sở lý thuyết (9 trang)

2.1. Phát hiện đối tượng; phân biệt classification, detection, segmentation.

2.2. Bounding box: xyxy, xywh và tọa độ chuẩn hóa; biến đổi hệ tọa độ khi resize/letterbox.

2.3. Nền tảng CNN, backbone, neck và detection head ở mức phục vụ các mô hình thực nghiệm.

2.4. Phát hiện một giai đoạn và hai giai đoạn: YOLO và Faster R-CNN.

2.5. Phát hiện theo từ vựng mở: YOLO-World, text prompt; phân biệt khả năng open-vocabulary và bằng chứng zero-shot.

2.6. IoU, NMS, ngưỡng confidence; tác động tới false positive/false negative.

2.7. Precision, Recall, F1, đường cong PR, AP50 và AP@[0.50:0.95]; macro/micro; xử lý crowd và maxDets theo COCO.

2.8. Lazy loading, streaming, batching và thiết kế adapter cho backend.

Hình dự kiến: minh họa IoU, sơ đồ một/hai giai đoạn, biến đổi letterbox. Công thức gắn với cách tính trong chương 5.

## Chương 3. Phân tích và thiết kế thư viện (8 trang)

3.1. Yêu cầu chức năng và phi chức năng; phạm vi đã được cài đặt.

3.2. Kiến trúc 12 mô-đun và các tầng xử lý; chỉ mô tả sâu các mô-đun liên quan câu hỏi nghiên cứu.

3.3. Giao diện công khai và hợp đồng dữ liệu: Box, Detection, Detections; class_id, tên lớp và tọa độ.

3.4. BaseModel, Detector, các adapter cụ thể; trách nhiệm dùng chung và trách nhiệm backend.

3.5. Luồng ảnh và video: media → preprocessing → models → outputs → postprocessing → visual.

3.6. Theo dõi phép biến đổi và khôi phục tọa độ; phạm vi khôi phục, crop và giới hạn biến đổi hình học.

3.7. Cấu hình, trạng thái mô hình, prompt được ghi nhớ, xử lý lỗi và tính độc lập của kết quả.

3.8. Thiết kế dữ liệu thực nghiệm và khả năng tái lập.

Hình dự kiến: sơ đồ kiến trúc, sơ đồ lớp, sequence diagram suy luận, sơ đồ vòng đời.

## Chương 4. Cài đặt (7 trang)

4.1. Công nghệ, phiên bản và cấu trúc mã; phân biệt phụ thuộc bắt buộc/tùy chọn.

4.2. Media lazy và pipeline tiền xử lý; một ví dụ xuyên suốt.

4.3. Cài đặt Detector: batch, stream, snapshot cấu hình, normalize kết quả.

4.4. Tích hợp Ultralytics và Torchvision; mô tả khả năng Hugging Face/Keras theo mã, chỉ gọi là đã kiểm chứng khi có kết quả chạy.

4.5. Hậu xử lý, NMS và trực quan hóa.

4.6. Bộ kiểm thử: dữ liệu tổng hợp, mock backend, trường hợp biên và mô hình thật; trích đoạn mã ngắn có phân tích.

## Chương 5. Thực nghiệm và đánh giá (12 trang)

5.1. Giả thuyết và thiết kế thực nghiệm được cố định trước khi xem kết quả.

5.2. Phần cứng, môi trường, checkpoint, tham số, seed, manifest dữ liệu.

5.3. Kết quả kiểm thử thư viện: tổng số chạy, pass/fail/error/skip và lý do skip; không coi test giả lập là kiểm chứng độ chính xác AI.

5.4. Chất lượng phát hiện trên cùng tập COCO validation: AP50, AP@[0.50:0.95], Precision/Recall/F1 tại ngưỡng cố định.

5.5. Thời gian xử lý: warm-up, đồng bộ GPU, batch=1, phân vị và phạm vi timer; độ biến thiên giữa các lần đo.

5.6. Phân tích chi phí lớp tích hợp: backend trực tiếp và Klygo, cùng trọng số/đầu vào/precision/cấu hình, kiểm tra đầu ra tương đương.

5.7. Phân tích lỗi: vật thể nhỏ, che khuất, nhầm lớp, bỏ sót, box trùng; cả ví dụ tốt và ví dụ sai.

5.8. Giới hạn: tập con, dữ liệu tiền huấn luyện, một máy đo, nhiệt độ/tải nền, cấu hình đầu vào khác nhau giữa kiến trúc, ngưỡng confidence không được hiệu chuẩn chung.

Bảng/hình dự kiến: phân bố dữ liệu, cấu hình mô hình, kết quả kiểm thử, AP, PR, latency, scatter chất lượng/thời gian, ảnh lỗi.

## Chương 6. Minh chứng sản phẩm và hướng dẫn sử dụng (3 trang)

6.1. Cài đặt và cấu hình môi trường.

6.2. Ví dụ chạy từ ảnh tới kết quả có nhãn, lưu và truy xuất dự đoán.

6.3. Thay backend qua giao diện chung; video/stream khi có kiểm chứng.

6.4. Tình huống lỗi thường gặp và cách tái chạy demo.

## Chương 7. Kết luận và hướng phát triển (3 trang)

7.1. Trả lời từng câu hỏi nghiên cứu bằng kết quả thực tế.

7.2. Đóng góp đạt được và giới hạn chưa giải quyết.

7.3. Hướng phát triển: mở rộng đánh giá, packaging, thêm adapter, tối ưu suy luận; tách khỏi chức năng hiện có.

## Phụ lục và tài liệu tham khảo

- Lệnh tái lập và requirements đã khóa phiên bản.
- Manifest dữ liệu, schema dự đoán COCO, cấu hình thí nghiệm.
- Kết quả test đầy đủ và chi tiết skip/failure nếu có.
- Bảng ánh xạ API/module tới ví dụ minh họa; tránh chép toàn bộ mã.
- Tài liệu gốc về COCO, Faster R-CNN, YOLO/YOLO-World và tài liệu chính thức của backend. Kiểm tra tác giả, năm và DOI/URL trước khi đưa vào BibTeX.

## Nguồn bằng chứng trong dự án

| Nội dung | Mã/tài liệu cần đối chiếu |
| --- | --- |
| Phụ thuộc và phiên bản Klygo | klygo/pyproject.toml |
| Giao diện công khai | klygo/klygo/__init__.py và __init__.py của từng module |
| Lifecycle, batching, streaming, benchmark | klygo/klygo/models/detection/base.py |
| YOLO và YOLO-World | klygo/klygo/models/detection/yolo.py, yolo_world.py |
| Torchvision | klygo/klygo/models/backend/torchvision.py |
| Kết quả chuẩn hóa | klygo/klygo/outputs/detect.py |
| Khôi phục tọa độ | klygo/klygo/preprocessing/trace.py |
| Hậu xử lý | klygo/klygo/postprocessing/detect.py |
| Kiểm thử tính đúng | klygo/test/ |
| Hạn chế workflow minh họa cũ | klygo/examples/integration/multi_backend_workflow.py |

## Điểm chờ chốt trước khi viết

- Danh sách thành viên đã được cung cấp; phân công cần rà soát theo công việc thực tế trước khi nộp.
- Abstract: đã chốt giữ bản tiếng Anh cùng Tóm tắt tiếng Việt.
- Quy mô thực nghiệm chính và danh sách mô hình cuối cùng: căn cứ pilot, không chọn theo mô hình có điểm cao.
