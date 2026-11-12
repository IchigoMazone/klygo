# Phần mở rộng mining và benchmark thực thi

Ngày: 06/11/2026. Người dùng cho phép thực hiện theo phương án phù hợp nhất sau khi thảo luận phạm vi.

- Giữ checkout nền tảng `0ca8a6a77600e1caf770ae8157d2df35360ee719` và kết quả ban đầu.
- Khảo sát snapshot `c4388f10bba7cf5ff02b9ebc4aa217b9f740e759` từ `origin/v2-release`, bổ sung mining (13 module).
- Không sửa mã thư viện để tạo kết quả thuận lợi; mọi script bổ sung nằm trong `report/experiments`.

## Mining

Chia thứ tự manifest 500 ảnh cố định thành train 300, validation 100, test 100. Loại annotation crowd, giữ box có diện tích dương sau clip. Mining chỉ dùng train. Đo chín phương pháp select cùng balance; 1 warm-up, 3 lượt đo CPU; tính điểm/cluster nằm trong timer. Export được ghép ảnh theo manifest và kiểm tra đọc lại số ảnh, box, nhãn. Feature vector có 726 chiều với 77 lớp hiện diện. Quota và Coverage dùng cùng hàm greedy; balance không tạo ảnh mới.

`Detections.export` xuất registry lớp hiện diện và đánh lại ID. Trước train, mọi nhãn được ánh xạ theo tên về cùng 80 lớp COCO bằng `prepare_training_canonical.py`. Lượt thử thiếu bước này được dừng và giữ riêng trong `extension/training`; không dùng trong báo cáo. Bằng chứng chính ở `extension/training_canonical`.

Fine-tune YOLOv8n: Full 300, Random/Weighted/Diversity 150 ảnh; giữ tập chọn seed 42; seed train 42/43/44, 5 epoch, batch 8, imgsz320, FP32, AdamW lr0=0.001, lrf=0.1, không warm-up/mosaic/mixup. Dùng last checkpoint, test riêng, không điều chỉnh siêu tham số theo test. Tất cả kết quả thấp hơn checkpoint ban đầu; báo cáo giữ nguyên kết quả âm và giới hạn thí nghiệm ngắn, không coi đây là xếp hạng phương pháp tổng quát.

## Chế độ thực thi

Hai checkpoint YOLOv8n/YOLO-World, cùng 100 JPEG, mỗi cặp model/chế độ trong tiến trình riêng, batch1 FP32 imgsz640 max_det100, 1 warm-up toàn tập và 3 lượt đo. Năm chế độ: Raw, Raw stream, Klygo native, Managed, Managed stream. Native không hỗ trợ invocation `stream=True` ở revision này. Timer tiêu thụ hết nguồn; serialize parity ngoài timer. Lấy mẫu RSS mỗi 10ms, VRAM là torch allocated peak, không phải tổng VRAM driver.

Thăm dò confidence0.001 dừng vì box suy biến bị validator native từ chối; dữ liệu Raw và log giữ trong `extension/execution`. Chạy lại toàn bộ ở confidence0.25, lưu `execution_conf025`, không loại ảnh. Không ghép số liệu hai cấu hình. Tất cả 800 cặp frame đối chiếu qua hai model đạt tiêu chí (tọa độ 0.01px, score0.001, cùng class/count). Các chế độ chạy nối tiếp, chưa randomize thứ tự; chưa có kiểm chứng stream dài hạn.

## Kiểm thử

502 ca của snapshot: 482 đạt, 20 skip, 0 lỗi/thất bại. Audit 13 module đạt. Lượt runner đầu có một assertion bắt UserWarning thất bại do policy warnings; runner được sửa để dùng `warnings='default'` như CLI unittest, không sửa test hay library. Log lượt đầu được giữ riêng. Kiểm tra lại SHA-256 của toàn bộ 500 ảnh đạt.

PDF trước mở rộng được giữ tại `output/pdf/bao_cao_klygo_truoc_mining.pdf`. Bản cập nhật giữ tên đầu ra chính và đóng gói lại LaTeX/bằng chứng; không cần chạy model để biên dịch.
