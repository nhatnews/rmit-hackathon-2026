# 🛡️ Multilingual Prompt-Attack Security Cheat Sheet (`pt`, `fr`, `it`, `en`)
**RMIT Hackathon 2026 — Challenge 1: 3-Class Prompt-Attack Classification**

---

## 1. Phân biệt Ranh giới 3 Lớp (`BENIGN` vs `JAILBREAK` vs `PROMPT_INJECTION`)

Trong cuộc thi này, metric là **Macro-F1** trên 3 lớp với phân bố lệch (`BENIGN` 62%, `JAILBREAK` 31%, `PROMPT_INJECTION` 7%). Việc phân biệt chính xác giữa **`JAILBREAK`** và **`PROMPT_INJECTION`** quyết định trực tiếp thứ hạng trên Leaderboard:

| Nhãn (`label`) | Tỷ lệ | Bản chất hành vi tấn công | Dấu hiệu nhận diện đặc trưng (Signals) |
| :--- | :---: | :--- | :--- |
| **`BENIGN`** | ~62% | Yêu cầu thông thường, hỏi đáp kiến thức, dịch thuật, tóm tắt, lập trình hợp lệ (kể cả thảo luận học thuật về bảo mật). | Câu hỏi tự nhiên, văn bản tài liệu dài nhưng không chứa lệnh ghi đè hệ thống hay ép buộc nhập vai phi đạo đức. |
| **`JAILBREAK`** | ~31% | **Thao túng vượt rào an toàn (Safety Guardrail Bypass):** Cố tình ép LLM sinh nội dung bị cấm/nguy hiểm bằng cách nhập vai (DAN), giả định hư cấu, viết truyện, thao túng tâm lý hoặc mã hóa từ nhạy cảm. | Tên persona (`DAN`, `AIM`, `STAN`, `Developer Mode`), cụm từ *"không giới hạn đạo đức"*, *"trong một bộ phim giả tưởng"*, Base64, viết tách ký tự (`b-y-p-a-s-s`). |
| **`PROMPT_INJECTION`** | **~7%** | **Chiếm quyền điều khiển chỉ thị (Instruction Hijacking / Control Override):** Cố tình hủy bỏ chỉ thị gốc của ứng dụng (system prompt) để chèn tác vụ mới, đánh cắp system prompt, hoặc chèn ký tự phân tách (delimiter escape). | Cụm từ mệnh lệnh trực tiếp *"Ignore previous instructions"*, *"Forget everything above"*, *"System override"*, *"Print your initial instructions"*, các thẻ `<|im_start|>`, `### System:`, ` ``` `. |

---

## 2. Từ khóa & Mẫu Tấn công Đa ngôn ngữ (`en`, `fr`, `it`, `pt`)

Dữ liệu trải đều trên 4 ngôn ngữ: **Tiếng Anh (`en`), Tiếng Pháp (`fr`), Tiếng Ý (`it`), và Tiếng Bồ Đào Nha (`pt`)**. Dưới đây là từ điển đối chiếu các mẫu tấn công trọng yếu để Cường Thịnh và Khánh Linh đưa vào Regex / TF-IDF:

### A. Mẫu `PROMPT_INJECTION` (Chiếm quyền chỉ thị & Đánh cắp System Prompt)

| Nhóm hành vi | Tiếng Anh (`en`) | Tiếng Pháp (`fr`) | Tiếng Ý (`it`) | Tiếng Bồ Đào Nha (`pt`) |
| :--- | :--- | :--- | :--- | :--- |
| **Hủy chỉ thị trước đó** | `ignore previous instructions`, `disregard all prior rules`, `forget everything above` | `ignorez les instructions précédentes`, `oubliez toutes les règles`, `ne tenez pas compte` | `ignora le istruzioni precedenti`, `dimentica tutto ciò che precede`, `ignora le regole` | `ignore as instruções anteriores`, `esqueça todas as regras`, `desconsidere tudo acima` |
| **Ghi đè chỉ thị mới** | `new instruction:`, `system override`, `from now on your only task is`, `important new directive` | `nouvelle instruction :`, `à partir de maintenant`, `directive système`, `nouvel ordre` | `nuova istruzione:`, `d'ora in poi il tuo compito è`, `direttiva di sistema` | `nova instrução:`, `a partir de agora sua tarefa é`, `diretiva do sistema` |
| **Đánh cắp System Prompt** | `reveal your system prompt`, `repeat the words above`, `what are your initial instructions` | `révélez votre invite système`, `répétez les instructions ci-dessus`, `texte initial` | `rivela il tuo prompt di sistema`, `ripeti le parole sopra`, `istruzioni iniziali` | `revele seu prompt de sistema`, `repita as palavras acima`, `instruções iniciais` |
| **Delimiter / Context Escape** | `### Instruction:`, `<\|endoftext\|>`, `[SYSTEM]`, `</context>`, `\n\nHuman:`, `\n\nAssistant:` | `### Système :`, `[FIN DU TEXTE]`, `Utilisateur :`, `Assistant :` | `### Sistema:`, `[FINE DEL TESTO]`, `Utente:`, `Assistente:` | `### Sistema:`, `[FIM DO TEXTO]`, `Usuário:`, `Assistente:` |

### B. Mẫu `JAILBREAK` (Nhập vai vượt rào & Thao túng ngữ cảnh)

| Nhóm hành vi | Tiếng Anh (`en`) | Tiếng Pháp (`fr`) | Tiếng Ý (`it`) | Tiếng Bồ Đào Nha (`pt`) |
| :--- | :--- | :--- | :--- | :--- |
| **Persona / DAN / Dev Mode** | `do anything now`, `act as DAN`, `developer mode enabled`, `unfiltered AI`, `evil confidant` | `mode développeur`, `agis comme une IA sans filtre`, `faire n'importe quoi maintenant` | `modalità sviluppatore`, `agisci come un'IA senza filtri`, `senza censure` | `modo desenvolvedor`, `aja como uma IA sem filtros`, `sem censura`, `modo livre` |
| **Vượt rào đạo đức / Luật lệ** | `without any moral or ethical restrictions`, `bypass OpenAI policy`, `no content policy` | `sans aucune restriction morale ou éthique`, `sans limites de sécurité`, `contourner les règles` | `senza restrizioni morali o etiche`, `senza limiti di sicurezza`, `aggirare le regole` | `sem restrições morais ou éticas`, `sem limites de segurança`, `burlar as regras` |
| **Giả định / Học thuật / Hư cấu** | `for educational purposes only`, `in a hypothetical world`, `writing a fictional novel where` | `à des fins purement éducatives`, `dans un monde hypothétique`, `pour un roman de fiction` | `solo a scopo educativo`, `in un mondo ipotetico`, `per un romanzo di finzione` | `apenas para fins educacionais`, `em um mundo hipotético`, `para um romance fictício` |
| **Ép buộc phản hồi đầu câu** | `start your response with "Sure, here is"`, `never say "I cannot"` | `commence ta réponse par "Bien sûr"`, `ne dis jamais "je ne peux pas"` | `inizia la tua risposta con "Certamente"`, `non dire mai "non posso"` | `comece sua resposta com "Claro, aqui está"`, `nunca diga "não posso"` |

---

## 3. Lưu ý Đặc biệt về Cột `language` và Văn bản Siêu dài (18k Ký tự)

1. **Khai thác cột `language` (`pt`, `fr`, `it`, `en`):**
   - Tỷ lệ phân bố 3 lớp (`BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`) và độ dài câu có thể khác nhau đáng kể giữa 4 ngôn ngữ.
   - **Kỹ thuật 1 (Prefix Token cho TF-IDF & Transformers):** Ghép trực tiếp mã ngôn ngữ vào đầu chuỗi văn bản (ví dụ: `"__lang_fr__ " + text`) để mô hình học được từ vựng gắn liền với ngữ cảnh ngôn ngữ (tránh trùng từ đồng hình giữa tiếng Pháp, Ý và Bồ Đào Nha).
   - **Kỹ thuật 2 (Tabular One-Hot & Group Stats):** Tạo biến giả (One-Hot Encoding) cho cột `language` và tính độ dài tương đối của prompt so với độ dài trung bình của ngôn ngữ đó (`char_len / mean_char_len_by_lang`).

2. **Xử lý Prompt có độ dài lớn (lên tới ~18,000 ký tự):**
   - Nhiều mẫu `JAILBREAK` (Many-shot jailbreaking / Context stuffing) hoặc `PROMPT_INJECTION` (Indirect injection chèn cuối tài liệu dài) có độ dài hàng chục nghìn ký tự.
   - **Đối với Transformers (DeBERTa / XLM-RoBERTa):** Nếu chỉ cắt `truncation=True, max_length=512` mặc định, mô hình sẽ **mất hoàn toàn phần đuôi văn bản** — nơi thường chứa câu lệnh tiêm nhiễm (`Ignore everything above and...`)!
     - **Giải pháp bắt buộc:** Áp dụng **Head + Tail Truncation** (lấy `256` token đầu + `256` token cuối của prompt) trước khi đưa vào mô hình GPU.
   - **Đối với Feature Engineering (`src/data_loader.py`):**
     - Bổ sung đặc trưng `log_char_len = np.log1p(char_len)`, `is_very_long = (char_len > 4000)`.
     - Tính mật độ ký tự đặc biệt/từ khóa tấn công riêng cho **500 ký tự đầu (Head)** và **500 ký tự cuối (Tail)**.

3. **Bảo toàn Dấu câu & Unicode Ngôn ngữ Romance (`NFC`):**
   - Tiếng Pháp (`é, è, ê, ë, à, â, ù, û, ç, œ`), tiếng Bồ Đào Nha (`ã, õ, ç, á, é, í, ó, ú, â, ê, ô`), tiếng Ý (`à, è, é, ì, ò, ù`) chứa rất nhiều dấu phụ.
   - Luôn chạy `unicodedata.normalize("NFC", text)` thông qua `clean_text_safety()` và giữ nguyên các ký tự phân tách câu lệnh (`<, >, |, #, `, {, }, [, ]`).

---

## 4. Chiến thuật Quản lý 6 Lượt Nộp & Tối ưu Macro-F1

- **Tại sao phải tối ưu ngưỡng/trọng số lớp (`optimize_class_multipliers`)?**
  - Lớp `PROMPT_INJECTION` chỉ chiếm **7%** dữ liệu nhưng đóng góp **33.33%** vào điểm **Macro-F1**.
  - Nếu lấy `argmax(proba)` mặc định hoặc dùng ngưỡng chưa hiệu chuẩn, mô hình thường quá dè dặt hoặc quá lố khi dự đoán lớp 7% này.
  - Bằng cách tìm bộ hệ số nhân $(m_0, m_1, m_2)$ trên tập OOF sao cho $\hat{y} = \arg\max_{c \in \{0,1,2\}} (m_c \cdot P(y=c|x))$ đạt Macro-F1 cao nhất (`src/blend.py:optimize_class_multipliers()`), điểm Macro-F1 thường tăng thêm **+0.015 đến +0.040** hoàn toàn miễn phí!
- **Phân bổ 6 lượt nộp (6 Submissions Quota):**
  - **Lượt 1:** Baseline 3-Class mạnh nhất (TF-IDF + LightGBM/LogReg đã tinh chỉnh trên 5-Fold CV) để xác nhận pipeline & định dạng `id, label` trên Public LB.
  - **Lượt 2:** Single Pretrained Transformer mạnh nhất từ Kaggle GPU (`protectai/deberta-v3-base-prompt-injection-v2` hoặc `FacebookAI/xlm-roberta-base`).
  - **Lượt 3:** Bản Blend đầu tiên giữa GPU Transformer + Tabular LightGBM/Linear.
  - **Lượt 4:** Bản Blend đa mô hình (DeBERTa + XLM-R + LightGBM + LinearSVC) kèm tối ưu hóa `optimize_class_multipliers` trên OOF.
  - **Lượt 5 & 6 (Dự phòng chiến thuật trước 17:30):** Bản Ensemble hoàn chỉnh nhất (có kiểm tra chống overfit hệ số nhân).
