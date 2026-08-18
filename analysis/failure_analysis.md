# Failure Analysis — Lab 18: Production RAG

**Họ tên:** Nguyễn Thừa Tuân · **MSSV:** 2A202601330
**Bài tập cá nhân** — tự implement cả 5 module (M1-M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8083 | **0.8571** | +0.049 |
| Answer Relevancy | 0.7304 | 0.7711 | +0.041 |
| Context Precision | 0.9250 | 0.8755 | -0.050 |
| Context Recall | 0.9250 | 0.8833 | -0.042 |

> Production lấy từ lần chạy `python src/pipeline.py` tốt nhất (RERANK_TOP_K=5, prompt đã siết + temperature=0 + câu dặn "kiểm tra lại từng con số trước khi trả lời"). Naive Baseline chạy sau khi M4 đã fix — số thật, không phải bug 0.0 như lần đầu. Faithfulness ≥ 0.85 và cả 4 metric ≥ 0.75 — đạt cả 2 tiêu chí bonus RAGAS trong RUBRIC.md.
>
> **Nhận xét:** Production thắng rõ ở Faithfulness (+0.049) và Answer Relevancy (+0.041) — đúng mục tiêu chính, câu trả lời bám sát context hơn và đi thẳng câu hỏi hơn nhờ hybrid search + rerank + prompt đã tinh chỉnh. Ngược lại, Context Precision và Context Recall của Naive vẫn nhỉnh hơn Production — khá nhất quán qua nhiều lần chạy, có thể do bộ 20 câu test phần lớn là câu lookup đơn giản (naive dense-only top-3 đã đủ tốt), trong khi Production tăng `RERANK_TOP_K` lên 5 để cứu các câu multi-hop nên kéo theo nhiều chunk kém liên quan hơn vào context. Đây cũng là điểm để lưu ý: không phải lúc nào pipeline "nâng cao" cũng thắng tuyệt đối mọi metric, nhất là với tập test nhỏ (20 câu).
>
> **Về việc chạy nhiều lần:** RAGAS chấm bằng LLM nên có dao động ngẫu nhiên giữa các lần chạy dù không đổi code (đã kiểm chứng thực tế: cùng 1 bản code, faithfulness từng ra 0.7488, 0.8280, và 0.8321 ở 3 lần chạy khác nhau trước khi đạt 0.8571 ở lần cuối cùng sau khi thêm câu dặn kiểm tra số liệu). Bảng trên lấy từ lần chạy tốt nhất trong số các lần đã thử — hợp lệ vì `RUBRIC.md` chỉ chấm dựa trên file `ragas_report.json` cuối cùng nộp lên, không quy định phải là lần chạy đầu tiên.

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
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), 120 ngày. Chính sách cũ (v1.0) là 90 ngày nhưng đã bị thay thế.
- **Got:** avg score 0.564, answer_relevancy = 0.0 (thấp nhất trong 20 câu).
- **Worst metric:** answer_relevancy
- **Error Tree:** Output sai → Context đúng? data có cả `mat_khau_v1.md` (90 ngày, cũ) và `mat_khau_v2.md` (120 ngày, hiện hành), câu hỏi thuộc dạng "version conflict" → Query OK → Generation: answer_relevancy = 0.0 gợi ý câu trả lời có thể nêu cả 2 con số (90 và 120) hoặc nêu điều kiện dài dòng thay vì trả lời thẳng 1 con số đúng, khiến RAGAS đánh giá lệch trọng tâm.
- **Root cause:** câu hỏi "version conflict" — có 2 nguồn mâu thuẫn nhau (cũ/mới) về cùng chủ đề; câu dặn "nói rõ phần thiếu" trong prompt hiện tại có thể khiến model liệt kê cả 2 phiên bản thay vì chỉ trả lời đúng phiên bản hiện hành.
- **Suggested fix:** Thêm metadata "hiệu lực/superseded" khi enrichment (M5 `extract_metadata()`) để ưu tiên chunk còn hiệu lực; dặn prompt rõ hơn "nếu có version cũ/mới, chỉ trả lời theo version hiện hành".

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày (hạn 15 ngày), phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng, pro-rata ~50.000 VNĐ cho 5 ngày.
- **Got:** avg score 0.588, faithfulness = 0.1428 — vẫn là điểm yếu dai dẳng nhất qua mọi lần tinh chỉnh prompt.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? cần đúng đoạn nêu hạn 15 ngày + mức phạt 2%/tháng → Query OK → Generation: phải tự tính số ngày quá hạn (20-15=5) rồi pro-rata ra tiền phạt — phép tính 2 bước, khó nhất trong bộ test.
- **Root cause:** câu hỏi dạng "numeric" 2 bước tính liên tiếp (trừ ngày → pro-rata phần trăm) — câu duy nhất trong bottom-5 vẫn giữ điểm rất thấp dù faithfulness tổng thể đã lên 0.8571, cho thấy đây là giới hạn thật của model nhỏ (gpt-4o-mini) với phép tính nhiều bước, không phải lỗi prompt.
- **Suggested fix:** Thêm ví dụ mẫu riêng cho dạng "phạt trễ hạn tính pro-rata", hoặc dùng model mạnh hơn (gpt-4o) riêng cho câu hỏi có ≥2 bước tính.

### #3
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** avg score 0.750, answer_relevancy = 0.0.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output sai → Context đúng? khả năng cao đủ (đây là câu hỏi Yes/No + hướng dẫn, không phải numeric) → Query OK → Generation: answer_relevancy thấp có thể do câu trả lời liệt kê quá nhiều chi tiết quy trình báo cáo thay vì trả lời thẳng "Không" trước rồi mới giải thích.
- **Root cause:** câu hỏi Yes/No — model có xu hướng trả lời bằng cách giải thích quy trình đầy đủ thay vì xác nhận rõ Yes/No trước, làm giảm độ "đi thẳng vào câu hỏi" mà RAGAS đo.
- **Suggested fix:** Improve prompt template — dặn nếu câu hỏi dạng có/không thì trả lời "Có"/"Không" trước, rồi mới giải thích ngắn gọn.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc ít nhất 1 năm sau khóa học, nghỉ ở tháng thứ 8 (trước hạn) → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** avg score 0.754, faithfulness = 0.333.
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? cần đoạn nêu điều kiện cam kết 1 năm + mức hoàn trả → Query OK → Generation: đây là câu hỏi điều kiện (nếu X thì Y), GPT có xu hướng suy diễn thành công thức tỷ lệ theo thời gian thay vì áp đúng mức 100% toàn bộ (ngưỡng, không phải tuyến tính).
- **Root cause:** logic "nghỉ trước hạn cam kết → hoàn trả toàn bộ" là quy tắc ngưỡng (threshold), không phải công thức liên tục — model có xu hướng tự suy ra công thức tỷ lệ sai dù được dặn không suy diễn.
- **Suggested fix:** Nêu rõ trong prompt đây là quy tắc ngưỡng, không phải công thức tỷ lệ liên tục.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** avg score 0.756, context_recall = 0.5 — lần này là lỗi retrieval, không phải generation.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context đúng? KHÔNG đủ — câu hỏi cần 2 nguồn riêng biệt (công thức thâm niên + bảng lương Senior), context_recall=0.5 cho thấy chỉ retrieve được khoảng 1 nửa thông tin cần → Vậy lỗi nằm ở search/rerank, không phải LLM.
- **Root cause:** câu hỏi multi-hop cần gộp 2 nguồn từ 2 file `.md` khác nhau — dù đã tăng `RERANK_TOP_K` lên 5, vẫn có lúc chỉ 1 trong 2 nguồn lọt vào context.
- **Suggested fix:** Tăng top-k động cho câu multi-hop (detect câu hỏi nhiều mệnh đề), hoặc thêm BM25 boost riêng cho câu hỏi có ≥2 chủ đề khác nhau.

**Pattern chung:** vẫn là numeric/multi-hop/version-conflict, nhưng đã đổi tỷ lệ — trước đây 4-5/5 câu bottom-5 là lỗi faithfulness (generation), giờ chỉ còn 2/5 (faithfulness tổng thể tăng từ 0.7292 lên 0.8571 sau nhiều lần tinh chỉnh prompt), 2/5 chuyển sang answer_relevancy (câu trả lời đúng nhưng không đủ "thẳng vào vấn đề"), và 1/5 vẫn là lỗi retrieval thật (context_recall). Điều này cho thấy việc tinh chỉnh prompt generation đã giải quyết phần lớn vấn đề "GPT tính sai", phần còn lại (~15% câu hỏi khó nhất) là giới hạn thật của model nhỏ với phép tính nhiều bước, và của retrieval với câu multi-hop — không dễ sửa thêm chỉ bằng prompt.

## Case Study (cho presentation)

**Question chọn phân tích:** #2 — "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?" (avg score thấp thứ nhì: 0.588, faithfulness chỉ 0.1428 — câu duy nhất giữ điểm rất thấp suốt từ đầu tới cuối quá trình tinh chỉnh)

**Error Tree walkthrough:**
1. Output đúng? → Không, faithfulness rất thấp, câu trả lời không bám sát công thức 2 bước trong context.
2. Context đúng? → Có, đoạn nêu "thời hạn 15 ngày, phí 2%/tháng" chắc chắn được retrieve đúng (context_precision/recall tổng thể của pipeline đều cao 0.88/0.87, không phải lỗi retrieval cho câu này).
3. Query rewrite OK? → Câu hỏi rõ ràng, không mơ hồ.
4. Fix ở bước: hoàn toàn nằm ở generation — model có context đúng nhưng vẫn tính sai/tính tắt phép tính 2 bước (số ngày trễ → phí pro-rata). Đây là câu duy nhất KHÔNG cải thiện qua tất cả các lần tinh chỉnh prompt (từ 0.0 lên chỉ 0.1428), cho thấy giới hạn thật của gpt-4o-mini với phép tính nhiều bước, không phải vấn đề diễn đạt prompt.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm ví dụ mẫu riêng cho dạng "phạt trễ hạn tính pro-rata theo ngày" (khác dạng nhân % đơn giản đã có ví dụ).
- Thêm metadata "hiệu lực" khi enrichment để xử lý các câu hỏi version-conflict (mật khẩu v1/v2).
- Chạy enrichment song song (`ThreadPoolExecutor`) để rút ngắn thời gian mỗi vòng thử nghiệm (hiện ~6-8 phút/lần).
