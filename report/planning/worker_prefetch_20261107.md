# Worker/prefetch và sáu mode

Người dùng đã chốt bổ sung ngày 07/11/2026: giải thích worker/prefetch có hình; giải thích sáu mode trước benchmark; đưa mining vào Bảng 3.2 và sơ đồ kiến trúc.

Revision giữ nguyên `c4388f10bba7cf5ff02b9ebc4aa217b9f740e759`. Không sửa core, không đổi checkpoint và không fine-tune thêm.

## Thiết kế đo

- Chạy mới cả sáu mode trong cùng phiên: Raw non-stream/stream, Native non-stream/backend-stream, Managed non-stream/stream. Native backend stream dùng `native=True, model_stream=True`; không truyền invocation `stream=True` cho native.
- 100 ảnh JPEG đầu theo manifest đã cố định, kiểm tra SHA-256 trước đo; batch1, FP32, imgsz640, confidence0.25, IoU0.7, max_det100.
- Mỗi model/mode/config trong tiến trình riêng; warm-up toàn tập một lượt; ba lượt timer; đối chiếu dự đoán bằng một lượt riêng ngoài timer. Không chạy các công việc GPU đồng thời.
- Managed stream: (W,P) = (0,0), (1,1), (2,1), (4,1), (2,2), (2,4). Khảo sát W ở P=1; khảo sát P ở W=2. Giữ batch cố định. Tái sử dụng (1,1) từ cùng phiên sáu mode, không lấy bảng cũ làm mốc.
- Đo thời gian toàn tập, throughput, thời gian tới kết quả đầu, RSS đỉnh mỗi 10ms và PyTorch allocated VRAM peak; lưu đủ từng lượt. Min--max không phải khoảng tin cậy.
- Kiểm chứng cùng source/checkpoint/config và parity từng frame (box/count/class, tọa độ0.01px, score0.001). Cấu hình có sai khác phải được ghi nhận.

## Vị trí bằng chứng

`extension/execution_six_20261107` chứa sáu mode; `extension/scheduling_20261107` chứa protocol được tạo trước khi đo và năm cấu hình bổ sung mỗi model. Script `run_scheduling_study.py` điều phối; `build_extension_assets.py` và `build_scheduling_assets.py` sinh bảng/hình từ JSON.

Các sơ đồ là mô hình giải thích, không phải số liệu: ba đường trách nhiệm và timeline D/I dùng khoảng thời gian giả định. Không coi số worker lớn hơn là hiệu suất tốt hơn trước khi có phép đo. Nguồn hữu hạn, một GPU, ba lượt và thứ tự tiến trình cố định chưa đủ cho khuyến nghị phổ quát hay cam kết bộ nhớ video dài hạn.
