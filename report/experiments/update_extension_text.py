"""Integrate extension sections without changing original experiment evidence."""
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]

def append(name, text):
    path = REPORT / "chapters" / name
    current = path.read_text(encoding="utf-8")
    if text.strip() not in current:
        path.write_text(current.rstrip()+"\n\n"+text+"\n", encoding="utf-8")

def replace(name, old, new):
    path = REPORT / "chapters" / name
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old,new), encoding="utf-8")

append("klygo02_theory.tex", r"\input{chapters/klygo02_mining}")
append("klygo03_design.tex", r"\input{chapters/klygo03_extension}")
append("klygo05_experiments.tex", r"\input{chapters/klygo05_extension}")
append("klygo01_intro.tex", r"""\section{Phạm vi mở rộng và câu hỏi nghiên cứu bổ sung}
Bên cạnh nền tảng 12 mô-đun tại revision \code{0ca8a6a}, báo cáo khảo sát revision \code{c4388f1} có 13 mô-đun, bổ sung mining và các chế độ native/managed. Các mô tả lifecycle, registry và kết quả kiểm thử ở phần nền tảng gắn với revision ban đầu; phần mở rộng nêu rõ thay đổi và được đo riêng. Không ghép số liệu hai revision thành một phép đo duy nhất.

\textbf{RQ5:} Các phương pháp mining làm thay đổi quy mô, phân bố nhãn, chi phí tạo dataset và kết quả fine-tune trên cùng tập test như thế nào?

\textbf{RQ6:} Trên cùng model và cấu hình, framework gốc, Klygo native và managed ở các chế độ stream/non-stream khác nhau thế nào về thời gian, bộ nhớ và tính tương đương dự đoán?

Thực nghiệm bổ sung có fine-tune ngắn trên các tập con với ba seed, không huấn luyện từ đầu. Quy mô này phục vụ kiểm chứng quy trình và phân tích thăm dò; không thay thế benchmark đầy đủ trên COCO train2017.""")
replace("klygo03_design.tex", "Thư viện xuất 12 mô-đun công khai.", "Revision nền tảng xuất 12 mô-đun công khai; revision mở rộng bổ sung mining thành mô-đun thứ 13 (Mục~\\ref{sec:extension-version}).")
replace("klygo00_front.tex", "Thư viện gồm 12 mô-đun công khai,", "Revision nền tảng có 12 mô-đun công khai, revision mở rộng thêm mining thành 13 mô-đun,")
replace("klygo00_front.tex", "Its twelve public modules cover", "The baseline revision has twelve public modules, while the extension adds mining as a thirteenth module. These modules cover")
replace("klygo00_front.tex", "\\textbf{Từ khóa:}", "Phần mở rộng kiểm chứng các phương pháp tuyển chọn dữ liệu, ghép ảnh với annotation đã xuất để huấn luyện, fine-tune YOLOv8n với ba seed và đo năm đường thực thi raw/native/managed. Mỗi nhóm kết quả ghi rõ revision, tập dữ liệu và giới hạn suy luận.\n\n\\textbf{Từ khóa:}")
replace("klygo00_front.tex", "\\textbf{Keywords:}", "The extension evaluates dataset curation and export assembly, short YOLOv8n fine-tuning with three seeds, and five raw/native/managed execution paths. Revision boundaries and the limited scope of the training study are explicitly documented.\n\n\\textbf{Keywords:}")
replace("klygo07_conclusion.tex", "Đề tài đã mô tả và kiểm chứng Klygo như một thư viện Python có 12 mô-đun công khai,", "Đề tài đã mô tả và kiểm chứng revision nền tảng của Klygo với 12 mô-đun và khảo sát revision mở rộng có thêm mining,")
replace("klygo07_conclusion.tex", "Nhóm chưa thực hiện huấn luyện, fine-tune, đo năng lượng hay thử tải phục vụ đồng thời.", "Phần mở rộng đã thực hiện fine-tune ngắn với ba seed; chưa huấn luyện từ đầu, đo năng lượng hay thử tải phục vụ đồng thời.")
replace("klygo07_conclusion.tex", "nguồn stream thư mục ảnh trong demo còn có frame index rỗng.", "nguồn stream thư mục ảnh trong demo revision nền tảng có frame index rỗng; phát hiện này không tự động áp dụng cho revision mở rộng.")
append("klygo07_conclusion.tex", r"""\section{Kết luận từ phần mở rộng}
\textbf{RQ5 -- Tuyển chọn dữ liệu.} Mining cung cấp chuỗi đọc, thống kê, chọn, cân bằng và so sánh bộ dữ liệu theo ảnh. Thực nghiệm kiểm chứng dữ liệu sau export và ghép ảnh có thể được đọc lại và dùng fine-tune. Số box và phân bố lớp thay đổi ngay cả khi cùng số ảnh. Kết quả train được báo cáo với ba seed và checkpoint ban đầu làm tham chiếu, không giả định mọi thao tác cân bằng hay phân cụm đều tăng AP.

\textbf{RQ6 -- Các chế độ thực thi.} Benchmark mới gọi framework gốc trực tiếp và phân biệt native với managed, bổ sung thời gian tới kết quả đầu cùng bộ nhớ. Kết quả chỉ áp dụng cho workload, revision và phần cứng đã đo; khác biệt container và media là một phần chi phí của giao diện công khai. Bằng chứng đối chiếu tọa độ, score và class được lưu riêng ngoài timer.

Hai giới hạn cài đặt đáng lưu ý là Quota/Coverage dùng chung thuật toán ở revision khảo sát, và export chỉ ghi annotation/manifest. Bước ghép ảnh, tạo YAML và huấn luyện do script thực nghiệm đảm nhiệm. Các điểm này giúp mô tả đúng năng lực thư viện, tránh biến tên API hoặc biểu đồ đẹp thành bằng chứng chất lượng mô hình.""")
append("klygo06_demo.tex", r"""\section{Tạo bộ dữ liệu bằng mining}
Ví dụ dành cho revision mở rộng, với dataset YOLO đã có ảnh và nhãn:
\begin{lstlisting}[caption={Chọn tập dữ liệu và xuất annotation cùng manifest}]
from klygo import datasets, mining
data = datasets.load("dataset/train", format="yolo")
summary = mining.stats(data)
selected = mining.select(data, method="weighted", limit=150)
mining.compare(data, selected).save("comparison.json")
selected.export("selected_labels", format="yolo")
\end{lstlisting}
Để train, script đọc manifest, sao chép ảnh nguồn sang thư mục images bằng tên tương ứng nhãn và tạo YAML trỏ tới các split. Chương trình \code{run\_mining\_extension.py} thực hiện bước ghép này, kiểm tra đọc lại và lưu ID được chọn. Chương trình \code{train\_mining\_extension.py} sử dụng bộ dữ liệu hoàn chỉnh đó, không giả định export tự sao chép ảnh.""")
append("klygo08_appendix.tex", r"""\section{Tái lập phần mở rộng}
Mã nguồn mở rộng được cố định bằng commit
\begin{center}\small\texttt{c4388f10bba7cf5ff02b9ebc4aa217b9f740e759}\end{center}
Snapshot nằm tại \code{tmp/klygo-c4388f1}; checkout nền tảng giữ nguyên. Tạo snapshot từ bản Git có commit này bằng \code{git archive}, giải nén vào thư mục trên, sau đó chạy theo thứ tự:
\begin{lstlisting}[language={},caption={Lệnh tái lập mining, huấn luyện và benchmark}]
python report/experiments/run_mining_extension.py
python report/experiments/benchmark_execution.py
python report/experiments/train_mining_extension.py
python report/experiments/check_extension.py
python report/experiments/build_extension_assets.py
\end{lstlisting}
Không chạy train đồng thời với benchmark suy luận. Dữ liệu, cấu hình, manifest tuyển chọn, bảng đo, dự đoán đối chiếu và kết quả theo seed được lưu dưới \code{klygo/artifacts/report/extension}. Script không ghi đè các kết quả mining đã hoàn tất; các lượt benchmark/train đã có JSON được bỏ qua khi tiếp tục. Muốn thực nghiệm cấu hình mới cần thư mục kết quả mới.

\input{generated/extension_checks}
""")
path = REPORT / "klygo_references.bib"
bib = path.read_text(encoding="utf-8")
if "opencvKmeans" not in bib:
    bib += r"""
@misc{opencvKmeans,
  author={{OpenCV}}, title={Clustering: kmeans and KmeansFlags},
  year={2025}, howpublished={\url{https://docs.opencv.org/4.12.0/d5/d38/group__core__cluster.html}},
  note={Tài liệu OpenCV 4.12.0, truy cập 06/11/2026}}
@misc{ultralyticsTrain,
  author={{Ultralytics}}, title={Model Training with Ultralytics YOLO},
  year={2026}, howpublished={\url{https://docs.ultralytics.com/modes/train/}},
  note={Tài liệu tham khảo truy cập 06/11/2026; thực nghiệm khóa phiên bản 8.3.228}}
"""
    path.write_text(bib, encoding="utf-8")
