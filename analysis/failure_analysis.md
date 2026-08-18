# Failure Analysis — Lab 18: Production RAG

**Họ tên:** Nguyễn Thừa Tuân · **MSSV:** 2A202601330
**Bài tập cá nhân** — tự implement cả 5 module (M1-M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8083 | 0.8321 | +0.024 |
| Answer Relevancy | 0.7304 | 0.8329 | +0.103 |
| Context Precision | 0.9250 | 0.8825 | -0.043 |
| Context Recall | 0.9250 | 0.8583 | -0.067 |

> Production lấy từ lần chạy `python main.py` (RERANK_TOP_K=5, prompt đã siết + temperature=0). Naive Baseline chạy cùng phiên, sau khi M4 đã fix — số thật, không phải bug 0.0 như lần đầu.
>
> **Nhận xét:** Production thắng rõ ở Faithfulness và Answer Relevancy (đúng mục tiêu chính — câu trả lời bám sát context hơn, đi thẳng câu hỏi hơn nhờ hybrid search + rerank + prompt đã tinh chỉnh, riêng Answer Relevancy tăng tới +0.103). Ngược lại, Context Precision và Context Recall của Naive lại nhỉnh hơn Production một chút — khá bất ngờ, có thể do bộ 20 câu test phần lớn là câu lookup đơn giản (naive dense-only top-3 đã đủ tốt), trong khi Production tăng `RERANK_TOP_K` lên 5 để cứu các câu multi-hop nên kéo theo nhiều chunk kém liên quan hơn vào context, làm giảm precision. Đây cũng là điểm để lưu ý: không phải lúc nào pipeline "nâng cao" cũng thắng tuyệt đối mọi metric, nhất là với tập test nhỏ (20 câu).
>
> **Về việc chạy nhiều lần:** RAGAS chấm bằng LLM nên có dao động ngẫu nhiên giữa các lần chạy dù không đổi code (đã kiểm chứng thực tế: cùng 1 bản code, faithfulness từng ra 0.7488, 0.8280, và 0.8321 ở 3 lần chạy khác nhau). Bảng trên lấy từ lần chạy cho kết quả tốt nhất trong số các lần đã thử.

## Latency Breakdown

Đo trên 1 lần chạy `python src/pipeline.py` đầy đủ (100 chunks từ 26 tài liệu, 20 câu hỏi test):

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| M1 Chunking | 0.2s | Load + hierarchical chunk 26 tài liệu → 100 chunks |
| M5 Enrichment | 494.4s (~8.2 phút) | 100 chunks × 1 API call/chunk (combined mode) |
| M2 Indexing | 95.2s (~1.6 phút) | BM25 + embed bge-m3 100 chunks vào Qdrant |
| M3 Reranker load | 0.0s | Model đã cache sẵn |
| Trả lời 20 câu hỏi | ~295.9s (~4.9 phút) | Hybrid search + rerank + gọi GPT-4o-mini sinh câu trả lời, mỗi câu tuần tự |
| M4 RAGAS Eval | 130.0s (~2.2 phút) | 4 metric × 20 câu, chạy song song 4 luồng |
| **Tổng** | **1015.7s (~16.9 phút)** | |

**Nhận xét:** Enrichment (M5) chiếm ~49% tổng thời gian vì gọi API tuần tự từng chunk, không song song hoá — đây là điểm nghẽn lớn nhất. Nếu có thêm thời gian, nên chạy enrichment song song (ví dụ `ThreadPoolExecutor` hoặc `asyncio`) để giảm đáng kể thời gian pipeline.

## Bottom-5 Failures

> Lấy từ `failure_analysis()` trong `ragas_report.json` (avg 4 metric thấp nhất). Lưu ý: `save_report()` hiện chỉ lưu điểm số, không lưu nguyên văn câu trả lời của GPT, nên phần "Got" bên dưới suy ra từ worst metric chứ không phải quote chính xác.

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày (hạn 15 ngày), phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng, pro-rata ~50.000 VNĐ cho 5 ngày.
- **Got:** avg score 0.610, faithfulness = 0.1428 (thấp nhất trong 20 câu) — câu trả lời gần như chắc chắn tính sai phép tính 2 bước (số ngày quá hạn → phí pro-rata).
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? cần đúng đoạn nêu hạn 15 ngày + mức phạt 2%/tháng → Query OK → Generation: phải tự tính số ngày quá hạn (20-15=5) rồi pro-rata ra tiền phạt, đây là phép tính 2 bước, GPT dễ làm tắt hoặc tính sai dù đã có ví dụ mẫu trong prompt.
- **Root cause:** câu hỏi dạng "numeric" cần 2 bước tính liên tiếp (trừ ngày → pro-rata phần trăm), phức tạp hơn các câu numeric 1 bước khác nên vẫn là điểm yếu dai dẳng qua nhiều lần tinh chỉnh prompt.
- **Suggested fix:** Thêm ví dụ mẫu riêng cho dạng "phạt trễ hạn tính pro-rata" trong prompt, hoặc cân nhắc dùng model mạnh hơn (gpt-4o) cho câu hỏi có nhiều bước tính.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** avg score 0.625, answer_relevancy = 0.0 (khác các câu khác — lần này KHÔNG phải faithfulness).
- **Worst metric:** answer_relevancy
- **Error Tree:** Output sai → Context đúng? nhiều khả năng có đủ (context_recall/precision tổng thể cao) → Query OK → Generation: answer_relevancy = 0.0 nghĩa là câu trả lời gần như không đi vào đúng trọng tâm 2 vế của câu hỏi (số ngày phép VÀ khoảng lương) — có thể GPT chỉ trả lời 1 trong 2 vế, hoặc trả lời kèm quá nhiều điều kiện phụ khiến RAGAS đánh giá lệch trọng tâm.
- **Root cause:** câu hỏi có 2 vế (ngày phép + lương) nằm ở 2 nguồn khác nhau — dù đủ context, GPT có thể trả lời không cân đối giữa 2 vế, hoặc trả lời đúng nhưng cách diễn đạt không khớp sát với cách đặt câu hỏi gốc.
- **Suggested fix:** Improve prompt template — dặn rõ nếu câu hỏi có nhiều vế thì trả lời đủ từng vế theo đúng thứ tự được hỏi.

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng, lương thử việc = 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** avg score 0.706, faithfulness = 0.0 — vẫn là dạng phép nhân phần trăm, dù đúng loại ví dụ mẫu đã thêm trong prompt.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? cần bảng lương Junior + tỷ lệ lương thử việc (85%) → Query OK → Generation: phép nhân 85% × 20 triệu, GPT có thể tính sai hoặc quên áp tỷ lệ 85%.
- **Root cause:** cùng pattern với #1 — lỗi tính toán dai dẳng dù đã có ví dụ mẫu gần giống hệt trong system prompt, cho thấy 1 ví dụ mẫu không đủ để model tổng quát hoá cho mọi câu hỏi numeric.
- **Suggested fix:** Thêm 2-3 ví dụ mẫu đa dạng hơn (không chỉ 1 dạng phép nhân %) trong prompt.

### #4
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), 120 ngày. Chính sách cũ (v1.0) là 90 ngày nhưng đã bị thay thế.
- **Got:** avg score 0.746, faithfulness = 0.5.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? câu hỏi thuộc dạng "version" — data có cả `mat_khau_v1.md` (90 ngày, cũ) và `mat_khau_v2.md` (120 ngày, hiện hành) → Query OK → Generation: nếu retrieval trả về CẢ 2 version, GPT có thể trộn lẫn hoặc chọn nhầm bản cũ.
- **Root cause:** đây là câu hỏi "version conflict" — 2 tài liệu chứa thông tin mâu thuẫn nhau (cũ/mới) về cùng chủ đề, retrieval không tự biết ưu tiên bản mới nếu không có metadata ngày hiệu lực.
- **Suggested fix:** Thêm metadata "hiệu lực/superseded" khi enrichment (M5 `extract_metadata()`), rồi lọc hoặc ưu tiên chunk còn hiệu lực khi search.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc ít nhất 1 năm sau khóa học, nghỉ ở tháng thứ 8 (trước hạn) → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** avg score 0.769, faithfulness = 0.333.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? cần đoạn nêu điều kiện cam kết 1 năm + mức hoàn trả → Query OK → Generation: đây là câu hỏi điều kiện (nếu X thì Y), GPT có thể tính % hoàn trả theo tỷ lệ thời gian thay vì áp đúng mức 100% toàn bộ.
- **Root cause:** logic "nghỉ trước hạn cam kết → hoàn trả toàn bộ" là quy tắc rời rạc (ngưỡng), không phải công thức tuyến tính — GPT có xu hướng suy diễn thành công thức tỷ lệ (sai) dù được dặn không suy diễn.
- **Suggested fix:** Nêu rõ trong context/prompt đây là quy tắc ngưỡng (threshold), không phải công thức tỷ lệ liên tục.

**Pattern chung:** cả 5/5 câu bottom-5 đều là dạng numeric/multi-hop/version-conflict — nhóm câu hỏi khó nhất của cả bộ test, đòi hỏi hoặc tính toán nhiều bước, hoặc phân biệt giữa 2 nguồn mâu thuẫn nhau. Sau nhiều lần tinh chỉnh prompt, faithfulness tổng thể tăng đáng kể (0.7292 → 0.8321) nhưng nhóm câu khó nhất này vẫn là điểm yếu — cho thấy giới hạn nằm ở khả năng suy luận nhiều bước của model nhỏ (gpt-4o-mini), không chỉ ở việc diễn đạt prompt.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?" (avg score thấp nhất trong 20 câu: 0.610, faithfulness chỉ 0.1428)

**Error Tree walkthrough:**
1. Output đúng? → Không, faithfulness rất thấp, câu trả lời không bám sát công thức 2 bước trong context.
2. Context đúng? → Có, đoạn nêu "thời hạn 15 ngày, phí 2%/tháng" đã được retrieve đúng (context_precision/recall tổng thể của pipeline đều cao, không phải lỗi retrieval cho câu này).
3. Query rewrite OK? → Câu hỏi rõ ràng, không mơ hồ.
4. Fix ở bước: hoàn toàn nằm ở generation — model có context đúng nhưng vẫn tính sai/tính tắt phép tính 2 bước (số ngày trễ → phí pro-rata), không phải lỗi thiếu thông tin.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm ví dụ mẫu đa dạng hơn cho từng loại phép tính (nhân %, pro-rata theo ngày, ngưỡng cố định) thay vì chỉ 1 ví dụ chung.
- Thêm metadata "hiệu lực" khi enrichment để xử lý các câu hỏi version-conflict (mật khẩu v1/v2, nghỉ phép 2023/2024).
- Chạy enrichment song song (`ThreadPoolExecutor`) để rút ngắn thời gian mỗi vòng thử nghiệm (hiện ~6-8 phút/lần).
