# Báo cáo Klygo

Đề tài: **Thiết kế và xây dựng thư viện Klygo hỗ trợ xử lý dữ liệu và đánh giá các mô hình phát hiện đối tượng**.

Học phần Nhập môn Trí tuệ nhân tạo, Đại học Phenikaa. Giảng viên: TS. Trần Đình Tân. Ngôn ngữ chính: tiếng Việt; giữ Abstract tiếng Anh theo mẫu.

## Tệp sử dụng

- `main.tex`: điểm vào của báo cáo mới.
- `chapters/klygo*.tex`: nội dung mới; các chương cũ được giữ để tham khảo, không được main.tex nạp.
- `klygo_references.bib`: tài liệu tham khảo mới.
- `generated/*.tex`: số liệu và bảng được tạo từ kết quả thực nghiệm.
- `images/klygo/`: hình mới; `images/logo.png` từ mẫu người dùng.
- `experiments/`: mã thực nghiệm và sinh biểu đồ.
- `template_original.zip`: bản lưu các tệp mẫu trước khi thay main.tex.
- `planning/`: đề cương và thiết kế trước thực nghiệm chính.

## Chỉnh thông tin nhóm

Tên và MSSV của ba thành viên được khai báo tại các macro `thanhVienMot`, `thanhVienHai`, `thanhVienBa` ở đầu main.tex và được dùng trên bìa. Bảng phân công trong `chapters/klygo00_front.tex` hiện là phương án nhiệm vụ; nhóm cần rà soát theo công việc thực tế trước khi nộp.

## Biên dịch

Chọn **XeLaTeX** trên Overleaf. Với MiKTeX/TeX Live cục bộ, chạy `powershell -NoProfile -File report/build.ps1` từ workspace hoặc bốn lệnh ở phụ lục. Font serif dùng Times New Roman khi có, nếu thiếu dùng TeX Gyre Termes. Cỡ chữ chính 13 pt, giãn dòng 1,2. Không cần bật shell escape.

Các bảng và hình đã được sinh sẵn, không cần chạy mô hình để biên dịch PDF. Nếu thay kết quả, chạy script `experiments/build_report_assets.py` sau khi kiểm tra hồ sơ dữ liệu.

## Phần mở rộng ngày 06/11/2026

Revision `c4388f10bba7cf5ff02b9ebc4aa217b9f740e759` của `origin/v2-release` có module mining và đường native/managed. Snapshot được lưu trong `tmp/klygo-c4388f1.zip`, giải nén tại `tmp/klygo-c4388f1`; các script mở rộng thêm đường dẫn này vào đầu `sys.path`, không thay checkout `klygo` hay bản editable đã dùng cho thí nghiệm nền tảng. Cả hai revision khai báo package 2.1.0, nên phải đối chiếu commit.

Thứ tự tái lập:

1. `python report/experiments/run_mining_extension.py` - 9 phương pháp chọn, balance, export annotation, ghép ảnh và kiểm tra đọc lại.
2. `python report/experiments/prepare_training_canonical.py` - ánh xạ class ID của từng export theo tên về cùng 80 lớp COCO. Đây là bước bắt buộc trước train.
3. `python report/experiments/run_scheduling_study.py` - sáu đường thực thi và các cấu hình worker/prefetch trên hai checkpoint, confidence 0,25, tiến trình riêng. Không chạy cùng train.
4. `python report/experiments/train_mining_extension.py` - fine-tune 5 epoch, 3 seed, Full/Random/Weighted/Diversity; đánh giá checkpoint cuối trên test riêng.
5. `python report/experiments/check_extension.py` - kiểm thử và audit snapshot.
6. `python report/experiments/build_extension_assets.py` rồi `python report/experiments/build_scheduling_assets.py` - tạo bảng/hình mới từ JSON, không thay số liệu nền tảng.

Sử dụng Python trong `.venv-report`. Kết quả mới nằm dưới `klygo/artifacts/report/extension`. Bản báo cáo hiện tại dùng `execution_six_20261107`, `scheduling_20261107` và `training_canonical`; phiên năm mode trước đó được giữ trong `execution_conf025`. Thư mục `execution` lưu lượt ngưỡng thấp có lỗi box suy biến, `training` lưu lượt thử có registry chưa thống nhất và **không được dùng kết luận**. Log các lượt được bảo tồn để truy nguyên. Dữ liệu mining chia 300/100/100 từ mẫu 500 ảnh COCO val2017: đây là nghiên cứu nhỏ về quy trình, không phải benchmark huấn luyện chuẩn trên toàn COCO. Full-data nghĩa là toàn bộ 300 ảnh train của nghiên cứu này.

`export()` ở revision mở rộng chỉ ghi annotation/manifest, không sao chép ảnh; script bổ sung bước ghép ảnh và YAML. Quota và Coverage hiện dùng chung thuật toán. Priority là tên tham số không prefix, không phải thứ tự ghi đè khi xung đột. Những giới hạn này được trình bày trong nội dung báo cáo.

## Sáu mode và worker/prefetch ngày 07/11/2026

`run_scheduling_study.py` chạy lại trọn bộ sáu mode Raw non-stream/stream, Native non-stream/backend-stream và Managed non-stream/stream trong một phiên mới. Kết quả chính nằm tại `extension/execution_six_20261107`; các bảng/biểu đồ sáu mode được sinh từ thư mục này, không ghép một hàng mới vào phép đo cũ. Native backend stream sử dụng `native=True, model_stream=True`; Klygo vẫn materialize generator trước khi trả kết quả ở commit khảo sát.

Phép đo Managed stream khảo sát (workers,prefetch) = (0,0), (1,1), (2,1), (4,1), (2,2), (2,4), giữ batch=1. Tham chiếu (1,1) lấy từ cùng phiên sáu mode; năm cặp còn lại nằm tại `extension/scheduling_20261107`. Mỗi cấu hình chạy trong tiến trình riêng, warm-up một lượt toàn bộ 100 ảnh, đo ba lượt, kiểm tra dự đoán ở lượt ngoài timer. Protocol được lưu trước khi đo. Không chạy train hoặc công việc GPU khác đồng thời.

Sau khi có đủ kết quả, chạy `build_extension_assets.py` rồi `build_scheduling_assets.py` để sinh các bảng và hình. `workers` là luồng giải mã/materialize của Klygo; `prefetch` là hàng đợi theo batch, không phải số GPU hoặc worker DataLoader huấn luyện. Báo cáo có sơ đồ luồng thực thi và timeline minh họa cơ chế; timeline không dùng số đo thực nghiệm. Mining đã được đưa vào Bảng 3.2 và tầng dữ liệu của sơ đồ kiến trúc, với ghi chú revision mở rộng.

## Căn cứ số liệu

- 500 ảnh chọn cố định từ COCO val2017, seed 20261105; 3.516 annotation, gồm 41 crowd, 3.475 GT hợp lệ và 79 lớp có hỗ trợ.
- YOLOv8n, YOLO-World v2 small, Faster R-CNN MobileNetV3-Large FPN; FP32, batch=1, GTX 1650 Ti Max-Q.
- AP tương ứng khoảng 38,24%, 39,07%, 25,08%; đây là chất lượng sau pipeline Klygo trên tập con.
- Thời gian chính: 100 ảnh, ba lượt, đo cặp `predict`/`forward` với cùng inference mode; đầu ra được kiểm tra tương đương.
- Suite gốc: 360 ca, 353 pass, 7 skip có lý do, 0 fail/error; ba test metric bổ sung được báo cáo riêng.

Bằng chứng gốc ở `../klygo/artifacts/report/`. Điểm pilot 16 ảnh không được dùng thay điểm chính. Thời gian ở `timing_controlled/` mới là đối chứng đã đồng nhất inference mode; các lần thử trước được giữ lại để truy vết.

Các giới hạn quan trọng được trình bày trong báo cáo: làm tròn confidence/tọa độ trong builder, restore tường minh sau tiền xử lý hình học ngoài model, khác biệt resize giữa kiến trúc, một tập con/một máy đo và các checkpoint chưa khảo sát. Thí nghiệm pretrained nền tảng không fine-tune; phần mining mở rộng có fine-tune riêng như mô tả bên trên.
