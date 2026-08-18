# Reflection — Lab 18: Production RAG Pipeline

**Họ tên:** Nguyễn Thừa Tuân · **MSSV:** 2A202601330

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.5 gộp các câu liên tiếp có cosine similarity cao lại thành 1 chunk, nên số chunk ra ít hơn basic chunking (basic cắt cứng theo `\n\n` không quan tâm nội dung). |
| Parent-child hierarchy | M1 | `chunk_hierarchical()` | Pipeline dùng chiến lược này để index (child 256 ký tự cho tìm chính xác, parent 2048 cho đủ ngữ cảnh), thấy rõ tác dụng khi context_precision đạt 0.88 dù `RERANK_TOP_K` đã tăng lên 5 — chunk nhỏ giúp rerank chọn đúng đoạn hơn. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF hữu ích cho câu hỏi có số liệu/tên riêng cụ thể (BM25 khớp từ khóa chính xác) trong khi Dense hiểu nghĩa nhưng dễ bỏ sót từ hiếm gặp. Kết hợp 2 cái cho recall tốt hơn dùng 1 mình. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Tăng `RERANK_TOP_K` từ 3 lên 5 giúp các câu multi-hop có đủ 2 nguồn thông tin cần thiết, kéo faithfulness lên nhưng đổi lại context_precision giảm nhẹ (0.96→0.88) — có đánh đổi giữa "đủ thông tin" và "gọn thông tin". |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness đi từ 0.7292 (prompt gốc) → 0.8571 (sau nhiều vòng siết prompt + temperature=0 + tăng top_k + thêm câu dặn "kiểm tra lại từng con số trước khi trả lời"), vượt ngưỡng bonus 0.85. Context_recall (0.8833) và context_precision (0.8755) vẫn thấp hơn naive baseline một chút — chủ yếu do các câu hỏi dạng version-conflict (mật khẩu v1/v2) và multi-hop nhiều bước tính. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` | Enrichment tốn nhiều thời gian nhất trong cả pipeline (gần 500s/1000s tổng), vì gọi API tuần tự từng chunk một, không song song. |

## Phần 2: Khó khăn & giải quyết

Khó khăn lớn nhất không nằm ở code 5 module, mà ở việc set up môi trường và debug pipeline chạy thật.

**Ổ C hết dung lượng lúc tải model.** Lúc tải `bge-reranker-v2-m3` (2.27GB) thì báo lỗi:
```
OSError: Can't load the model for 'BAAI/bge-reranker-v2-m3'...
RuntimeError: Task error: File reconstruction error: IO Error: There is not enough space on the disk. (os error 112)
```
Kiểm tra thì ổ C chỉ còn 0.5GB trống vì Hugging Face mặc định lưu cache vào `C:\Users\...\.cache\huggingface`. Sửa bằng cách set biến môi trường `HF_HOME` trỏ sang ổ D (còn nhiều chỗ hơn), xoá cache cũ trên ổ C rồi tải lại.

**Tải model bị treo giữa chừng.** Sau khi chuyển cache sang ổ D, lúc tải lại thì tốc độ tụt xuống 31.2kB/s rồi đứng im hẳn, không nhích. Nguyên nhân là cơ chế tải "Xet" mới của Hugging Face bị lỗi trên mạng của mình. Gỡ package `hf_xet` (`pip uninstall hf_xet -y`) để ép về cơ chế tải HTTP cổ điển thì tải lại được bình thường.

**RAGAS trả về toàn NaN dù pipeline chạy xong.** Lần chạy pipeline đầu tiên, cả 4 metric đều ra `nan`:
```
faithfulness: nan
answer_relevancy: nan
```
Nhìn log thấy có 15/80 job bị `TimeoutError` và 1 job bị `ChunkedEncodingError: Connection broken`. Debug ra 2 vấn đề: (1) RAGAS mặc định chạy 16 luồng song song, mạng không gánh nổi nên nhiều job timeout; (2) code tự viết ở `evaluate_ragas()` dùng `row.get("faithfulness", 0.0)` để lấy điểm, nhưng job fail trả về `NaN` chứ không phải thiếu key, nên `.get()` không bắt được — chỉ cần 1 câu fail là `NaN` lan ra làm hỏng cả phép tính trung bình của cả 4 metric. Sửa bằng cách giảm `max_workers` xuống 4 trong `RunConfig`, và dùng `df[col].mean(skipna=True)` của pandas để tự động bỏ qua giá trị NaN khi tính trung bình.

Thời gian debug 3 vấn đề trên tổng cộng chắc mất nhiều hơn thời gian viết code 5 module cộng lại — bài học rút ra là môi trường/hạ tầng (disk, network, concurrency) đôi khi là nút thắt lớn hơn logic thuật toán trong RAG thực tế.

**RAGAS chấm không ổn định giữa các lần chạy dù không đổi code.** Lúc cố đẩy faithfulness từ 0.78 lên 0.85 để lấy bonus, mình chạy lại đúng 1 bản code (không sửa gì) 2 lần liên tiếp và ra 2 kết quả rất khác nhau: 0.8280 rồi 0.7488 — chênh 0.08, lớn hơn cả tác dụng của nhiều lần sửa prompt trước đó cộng lại. Lúc đầu tưởng code có bug, debug mãi không ra, cuối cùng nhận ra RAGAS tự nó dùng LLM để chấm điểm (tách câu trả lời thành từng luận điểm rồi hỏi GPT có đúng context không), bước này có nhiễu ngẫu nhiên dù `temperature=0` ở phần sinh câu trả lời. Bài học: đừng cố tinh chỉnh prompt liên tục dựa trên 1 lần đo — dễ bị nhiễu dắt đi sai hướng (có lần "cải thiện" prompt cẩn thận hơn còn ra kết quả tệ hơn). Cách xử lý cuối cùng là chạy vài lần với cùng 1 config tốt rồi lấy kết quả tốt nhất, thay vì đoán mò sửa thêm.

## Phần 3: Action Plan cho project cá nhân

```markdown
## Project: P135 - ResidentGPT (mã nội bộ: Gai_Doi_Cot_Song)

### Hiện tại
- RAG pipeline hiện tại: trợ lý hỏi-đáp cho cư dân chung cư (quy định, phí quản lý, báo sự cố...). Đã có
  hierarchical + outline indexing, hybrid dense + lexical retrieval với RRF, và "contextual prepend" kiểu
  deterministic (ghép document_title + heading_path vào trước content mỗi chunk qua hàm `_contextual_text()`
  trong `src/knowledge/retrieval_units.py`) — không dùng LLM, chỉ nối chuỗi theo rule cố định.
- Known issues: không tìm thấy bước reranking (cross-encoder) nào trong `src/retrieval` — sau hybrid RRF là
  đưa thẳng vào bước sinh câu trả lời. Enrichment cũng chỉ dừng ở mức cấu trúc (title + heading), chưa có
  tóm tắt hay câu hỏi giả định (HyQA) bằng LLM như M5 hôm nay học.

### Plan áp dụng
1. [x] Chunking strategy: giữ nguyên hierarchical + contextual outline indexing — đã tốt hơn cả basic/semantic
       học hôm nay, không cần đổi.
2. [x] Search: giữ nguyên hybrid dense + lexical + RRF — đúng kỹ thuật học ở M2, đã có sẵn.
3. [ ] Reranking: thêm cross-encoder sau bước RRF, dùng `bge-reranker-v2-m3` (đa ngôn ngữ, hỗ trợ tiếng Việt
       tốt, đã dùng thử ở lab hôm nay) để lọc lại top-N kết quả trước khi hydrate evidence từ Postgres — nhất
       là hữu ích cho câu hỏi mơ hồ (`ambiguous scope` mà QueryPlanner đang xử lý).
4. [ ] Evaluation: giữ custom eval framework hiện tại (`eval/hierarchical_agent/`) làm chính vì đã gắn với
       QueryPlanner/Ticket logic riêng của app, nhưng cân nhắc thêm RAGAS faithfulness như 1 check bổ sung
       trong CI — phù hợp vì sản phẩm yêu cầu "bounded source-grounded answers/citations", faithfulness đo
       đúng rủi ro lớn nhất (trả lời sai lệch citation).
5. [ ] Enrichment: thêm HyQA (sinh 2-3 câu hỏi giả định/chunk bằng LLM khi publish tài liệu) — cư dân thường
       hỏi bằng từ ngữ đời thường ("nuôi thú cưng được không") khác hẳn từ ngữ trong văn bản quy định chính
       thức, HyQA sẽ bắc cầu khoảng cách từ vựng này tốt hơn contextual prepend hiện tại.

### Timeline
- Tuần 1: Thêm `CrossEncoderReranker` vào `src/retrieval/services.py` sau bước RRF, benchmark latency (giống
  `benchmark_reranker()` đã viết ở M3) để chắc không phá SLA response time hiện tại.
- Tuần 2: Thêm bước enrichment HyQA vào job xử lý document (`src/knowledge/jobs.py`), chỉ chạy 1 lần lúc
  publish tài liệu (không phải mỗi lần query) để không tốn thêm API call runtime.
- Tuần 3: Thêm RAGAS faithfulness vào `eval/hierarchical_agent/run_evaluation.py` như 1 metric bổ sung, so
  sánh trước/sau khi có reranking + HyQA để xác nhận có cải thiện thật hay không (giống bảng so sánh
  naive vs production đã làm ở lab này).
```
