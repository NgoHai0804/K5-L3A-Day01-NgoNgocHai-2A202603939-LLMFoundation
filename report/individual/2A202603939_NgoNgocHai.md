# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603939
- **Họ và tên:** Ngô Ngọc Hải
- **SEED:** 2559

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có output | Có |
| [`lab1_report.json`](lab1_report.json) — báo cáo cuối notebook | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay | Có |

## 2. Kết quả chính

Mình tổng hợp lại kết quả trong file JSON sau khi chạy notebook. Nhìn vào bảng, MLP cho F1 test cao hơn LogReg, còn các model embedding có F1 SUB tốt nhưng thời gian xử lý cũng lâu hơn.

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.9213 | 0.8889 | 0.105 | 0.000 |
| BoW + MLP | 0.9514 | 0.9189 | 1.424 | 0.000 |
| MiniLM + kNN | 0.9137 | 0.9500 | 64.922 | 0.000 |
| MiniLM + LogReg | 0.9375 | 0.9744 | 57.075 | 0.000 |
| MiniLM + MLP head | 0.9574 | 0.9744 | 63.971 | 0.000 |
| LLM nhỏ zero-shot | 0.9048 | 0.9048 | 2529.889 | 0.000* |
| LLM lớn zero-shot | 0.9756 | 0.9756 | 2389.987 | 0.000* |
| LLM nhỏ few-shot | 0.9268 | 0.9268 | 2656.751 | 0.000* |

`*` Chi phí LLM trong JSON hiện là 0 USD/1k tin. Mình không xem đây là giá API thực tế, vì có thể phần tính chi phí chưa được cấu hình trong lần chạy này.

## 3. Dự đoán và thực tế

File JSON chỉ lưu phần mình dự đoán trước, không lưu kết quả dự đoán của từng tin. Vì vậy mình không thể chọn chính xác ba dự đoán lệch nhất. Dưới đây là vài chỗ có thể so sánh với số liệu đang có:

| Mục dự đoán | Dự đoán | Kết quả đối chiếu |
| --- | --- | --- |
| `ml_f1` | F1 khoảng 0.90, recall cao hơn precision | F1 thực tế là 0.9213, khá gần dự đoán. Nhưng mình đoán sai về precision và recall: precision là 1.00, cao hơn recall 0.8542. |
| `dl_relu` | Bỏ ReLU thì F1 gần LogReg | MLP đạt 0.9514, cao hơn LogReg ở mức 0.9213. Dù vậy JSON không nói rõ lần chạy kết quả có bỏ ReLU hay không, nên mình chưa thể kết luận đây là so sánh trực tiếp của dự đoán đó. |
| `hard_eval` | BoW khoảng 50–60% | Trong bảng output nhóm hiện có, BoW + LogReg chỉ đúng 10/50 case. Kết quả này thấp hơn nhiều so với dự đoán ban đầu. |

## 4. Phân tích 50 case khó

Nhóm khó nhất với BoW + LogReg là `ham_looks_spam`: cả 10 tin ham trong nhóm đều bị nhận nhầm thành spam. Ngược lại, model làm khá tốt ở `manipulation` (8/8), `natural_spam` (10/10), `obfuscation` (6/6) và `prompt_injection` (5/6). Vậy là mình đoán nhầm khi cho rằng obfuscation sẽ là nhóm BoW kém nhất.

Hai ví dụ cho thấy vì sao nhóm này dễ gây nhầm:

- “FREE pizza in the office kitchen right now, come quick before it's gone” là tin ham nhưng chứa `FREE`, `right now` và lời thúc giục. BoW dựa vào từ khóa bề mặt nên có thể gán spam dù ngữ cảnh là chia sẻ pizza nội bộ.
- “URGENT: mum's car broke down near the station, can you pick her up? Call me asap” là tin ham khẩn cấp thực sự. Các từ `URGENT`, `Call`, `asap` giống mẫu spam nên mô hình từ khóa khó phân biệt ngữ cảnh cá nhân.

Hai câu này nằm trong bộ case khó và được gán nhãn ham. File kết quả không lưu dự đoán theo từng ID, nên mình không khẳng định đây chính xác là hai câu model đã phân loại sai; chúng minh họa kiểu từ ngữ dễ khiến BoW nhầm.

## 5. Thử thách cá nhân

SEED của mình là 2559, tương ứng thử thách chạy kNN với k = 1, 5, 25. Trong notebook, phần kết quả ghi đây là **weighted F1**: lần lượt 0.9770, 0.9783 và 0.9672. F1 cao nhất ở k = 5; khi k tăng lên 25, điểm giảm, có thể do các láng giềng ham đông hơn lấn át một số tin spam. JSON cũng ghi `challenge: 3`, tức đề thứ tư, khớp với thử thách này.

## 6. Phương án cho sản phẩm thật

Nếu phải chọn từ kết quả hiện có, mình sẽ bắt đầu với **BoW + MLP**. Model này có p95 1.424 ms/tin, thấp hơn giới hạn 50 ms khá nhiều; F1 test là 0.9514 và F1 SUB là 0.9189. Mười triệu tin mỗi ngày tương đương trung bình khoảng 116 tin/giây, nhưng vẫn cần thử tải thực tế để kiểm tra giờ cao điểm, vì benchmark notebook chưa chứng minh hệ thống chịu được tải production.

Đổi lại, MLP không có F1 cao nhất và BoW vẫn dễ nhầm tin nhắn bình thường nếu có các từ như “FREE” hoặc “URGENT”. LLM lớn có F1 cao nhất (0.9756), nhưng p95 khoảng 2.39 giây nên không đáp ứng giới hạn 50 ms; chi phí trong JSON cũng chưa đáng tin để dự toán. Mình sẽ cân nhắc dùng LLM để xem lại các mẫu khó, thay vì gọi nó cho từng tin nhắn.

## 7. AI log và điều đã học

Mình đã dùng AI để hỗ trợ rà lại báo cáo và đối chiếu số liệu, tên artifact với hướng dẫn nộp bài. AI chỉ ra một số chỗ cần kiểm tra; mình đã đối chiếu lại notebook và sửa phần tên ảnh cùng mô tả thử thách. Mình cần bổ sung thêm các lần dùng AI khác trong lúc làm lab nếu có, gồm câu hỏi, câu trả lời và phần mình tự sửa.

Qua bài này mình hiểu rõ hơn rằng điểm tốt trên tập test chưa chắc có nghĩa là model hiểu đúng ngữ cảnh. BoW có thể bắt được nhiều từ khóa spam, nhưng cũng vì thế mà nhầm tin nhắn bình thường có những từ như “FREE” hay “URGENT”. Khi chọn model cho sản phẩm, mình cần nhìn cả F1, tốc độ và chi phí; chỉ nhìn một con số thì chưa đủ.
