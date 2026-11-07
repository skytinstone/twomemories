# 전제 확인 기록 (assumptions)

## 환경과 버전 (2026-11-07, 로컬 확인)

| 항목 | 값 |
|---|---|
| 머신 | MacBook Pro 14, Apple M3 Pro, macOS (Darwin 24.4.0) |
| Python | 3.11.13 (`.venv`, uv) |
| torch | 2.15.0 (MPS 사용 가능) |
| transformers | 5.22.2 |
| datasets | 5.1.0 |
| hybridquant | 0.0.1, `~/workspace/gemini3/hybridquant` 커밋 405a86a, `pip install -e`로 설치. import 이름 `hybridquant` (pyproject `name = "hybridquant"`, `src/hybridquant`) |
| 모델 | `Qwen/Qwen3.5-0.8B` (Base가 아닌 post-trained 체크포인트. HF에 `-Instruct` 접미사 레포는 없음). `architectures = ["Qwen3_5ForConditionalGeneration"]`, `config.json`에 `vision_config` 있음 |
| 본 실험 모델 | `Qwen/Qwen3.5-4B` (같은 아키텍처, `vision_config` 있음 확인), 규모 실험 후보 `Qwen/Qwen3.5-9B` |
| GDN 커널 | 로컬은 `flash-linear-attention`, `causal_conv1d` 미설치 → transformers의 PyTorch 참조 구현(`torch_chunk_gated_delta_rule`)으로 동작 (실행 로그의 fallback 경고로 확인) |

파일 경로는 `.venv/lib/python3.11/site-packages/transformers/` 기준이다. `M` = `models/qwen3_5/modeling_qwen3_5.py`, `C` = `cache_utils.py`.

## 전제 1. 시각 특징 주입 위치

**결론: 입력 임베딩층에서만 주입된다(DeepStack류 중간 층 주입 없음).** 확인됨.

근거:
- `M:1621` `Qwen3_5Model.forward`에서 이미지 특징은 입력 임베딩의 자리 토큰 위치에만 들어간다.
  ```python
  image_mask, _ = self.get_placeholder_mask(
      input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
  )
  inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
  ```
- `M:1278-1279` `Qwen3_5TextModel.forward`의 decoder 루프는 hidden_states만 넘기며 시각 특징 인자(`visual_pos_masks`, `deepstack_visual_embeds`)가 없다.
  ```python
  for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
      hidden_states = decoder_layer(
          hidden_states,
          position_embeddings=position_embeddings,
  ```
- `M` 전체에 `deepstack` 문자열이 없다(`grep -n deepstack` 결과 0줄). 0.8B `config.json`의 `vision_config.deepstack_visual_indexes`는 `[]`이며 modeling 코드가 읽지 않는다.

설계 영향: A4의 "중간 층 주입" 분기는 해당 없음. 개입 정의 그대로. runner는 결과 줄 `meta.inject_layers = [0]`을 기록한다(0 = 입력 임베딩). 층별 분석을 할 때 별도 주의 표기는 필요 없다. 다만 4B/9B는 config를 따로 보지 않았으므로 Colab preflight에서 `deepstack_visual_indexes`를 다시 확인한다(UNVERIFIED for 4B/9B config 값; 코드 경로는 동일 클래스).

## 전제 2. 이미지 토큰 span

**결론: 이미지 자리 토큰 id = `config.image_token_id` = 248056 (`<|image_pad|>`). 프롬프트에서 연속 span이며 길이 = prod(image_grid_thw) / spatial_merge_size² 이다.** 확인됨.

근거:
- `M:1508` 자리 마스크는 `input_ids == image_token_id`.
  ```python
  else:
      special_image_mask = input_ids == self.config.image_token_id
      special_video_mask = input_ids == self.config.video_token_id
  ```
- `M:1482` 이미지 특징 개수 = `prod(grid_thw) // merge**2`, 그리고 `get_placeholder_mask`가 토큰 수와 특징 수가 같은지 검사한다.
  ```python
  vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
  split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
  vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
  ```
- 실측(0.8B processor, 448×448 이미지): `image_grid_thw = [[1, 28, 28]]`, 자리 토큰 196개 = 784/4, span `[4, 200)`, 전체 218토큰. 템플릿 결과: `<|im_start|>user\n<|vision_start|><|image_pad|>×196<|vision_end|>질문<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (기본 chat template이 빈 think 블록을 붙인다).
- M-RoPE(`M:1366` `get_rope_index`): 이미지 토큰은 (t, h, w) 3축 위치, 이미지 뒤 텍스트는 `max(h, w) // merge`만큼 건너뛴 위치에서 이어진다. 실측: 이미지 직전 토큰 위치 3, 첫 이미지 토큰 (4, 4, 4), 마지막 텍스트 토큰 35 (T=218).

설계 영향: `model.encode`가 마스크로 span을 찾고 연속이 아니면 예외, n_vis == prod/merge² 를 assert한다. 위치는 `get_rope_index`로 한 번 계산해 두고 슬라이스해 넘긴다(`position_ids`를 직접 넘김). 토큰 삭제 시 남은 시각 토큰은 원래 M-RoPE 위치를 유지한다(이것이 "토큰 삭제"의 정의). position_ids의 첫 행(text row)은 causal mask용이므로 실제 캐시 내 순번(0..T'-1)으로 다시 만든다.

## 전제 3. state dtype

**결론: recurrent state는 fp32로 저장된다. config의 `text_config.mamba_ssm_dtype = "float32"`는 modeling 코드가 읽지 않는다(값과 실제 dtype이 일치할 뿐 원인이 아님).** 로컬 PyTorch 경로에서 확인됨. GPU의 fla 커널 경로는 UNVERIFIED.

근거:
- `M:325` chunk 경로는 q/k/v/beta/decay를 fp32로 올린 뒤 계산하고, `M:393` 초기 state도 같은 dtype(fp32)으로 만든다.
  ```python
  query, key, value, beta, decay = [
      x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
      for x in (query, key, value, beta, decay)
  ]
  ```
- `C:1186`, `C:1245` 캐시는 처음 받은 텐서의 dtype으로 버퍼를 만들고 이후 `copy_`로 덮어쓴다.
  ```python
  self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
  ...
  self.recurrent_states[state_idx].copy_(recurrent_states)
  ```
- `mamba_ssm_dtype`: `models/qwen3_5/` 아래 어떤 파일에도 등장하지 않는다(`grep -rn` 0줄).
- 실측(0.8B, MPS bf16 모델): `recurrent_states[0]` shape `[1, 16, 128, 128]`, dtype `torch.float32`; conv state `[1, 6144, 4]` bf16; KV `[1, 2, T, 256]` bf16.

설계 영향: budget에서 state full = 32bit, KV full = 16bit로 계산한다. 양자화는 hybridquant `fake_quant`가 fp32로 계산 후 원 dtype(fp32)으로 돌려준다. Colab에서 fla 커널이 설치돼 있으면 state dtype이 달라질 수 있으므로 preflight 결과의 state dtype을 확인한다(colab_setup은 `USE_HUB_KERNELS=NO`로 hub 커널 교체를 끈다).

## 전제 4. 캐시 클래스

**결론: VL 모델도 `DynamicCache(config=...)`를 쓰고, 층별로 full_attention → `DynamicLayer`, linear_attention → `LinearAttentionLayer`가 만들어진다. hybridquant `hooks.py`가 패치하는 `DynamicLayer.update`와 `LinearAttentionLayer.update_recurrent_state`가 그대로 있다.** 확인됨.

근거:
- `C:1376-1385` 층 타입 → 캐시 층 클래스 매핑.
  ```python
  DYNAMIC_LAYER_TYPE_MAPPING = {
      "full_attention": DynamicLayer,
      ...
      "linear_attention": LinearAttentionLayer,
  ```
- `C:118` `DynamicLayer.update(self, key_states, value_states, *args, **kwargs)` → `C:135` `torch.cat`으로 새 토큰을 붙인다. `C:1232` `LinearAttentionLayer.update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs)`.
- hybridquant `hooks.py:140-141` 이 두 메서드를 저장 후 `:186` 등에서 교체한다. `M:641` GDN이 `cache_params.update_recurrent_state(...)`를 호출하고, attention은 `past_key_values.update(...)`를 호출한다.
- 실측: `make_cache(model)` 결과 `DynamicCache`, `layers[:4] = [LinearAttentionLayer ×3, DynamicLayer]`. 훅을 건 full 조건에서 `state_calls`, `kv_calls`가 0보다 크다(L1 결과 참조).

설계 영향: 어댑터 불필요. `quant.py`는 hybridquant의 `quantized_cache`, `fake_quant`, `QuantSpec`, `make_cache`, `budget.cache_sizes`를 import만 한다.

## 전제 5. 청크 prefill

**결론: GDN forward는 내부에서 64토큰 청크로 계산하지만(`chunk_size=64`), state는 forward 호출 끝에 층마다 1회만 캐시에 쓴다. 청크 경계마다 쓰지 않는다.** 코드와 실측으로 확인됨.

근거:
- `M:293`, `M:404-411` 청크 루프는 지역 변수 `last_recurrent_state`만 갱신한다.
  ```python
  for i in range(num_chunks):
      v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
      ...
      last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
  ```
- `M:626-641` 루프가 끝난 뒤 한 번 캐시에 쓴다.
  ```python
  core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
      ...
  )
  # Update cache
  if cache_params is not None:
      cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
  ```
- 실측(0.8B, 218토큰 단일 forward): `state_calls = 18` = GDN 층 수, `kv_calls = 6` = attention 층 수. 2단 prefill은 36, 시각 토큰 64개씩 4청크로 나눈 6회 forward는 108 = 18×6. 즉 쓰기 횟수는 토큰 수가 아니라 forward 호출 수에 비례한다.

설계 영향(A4 규칙 "시퀀스 끝 1회 쓰기" 적용): `8ef`/`6ef`는 시각 토큰을 256개씩 잘라 forward를 반복 호출하고, 매 청크 끝에서 모든 GDN 층의 state를 Q(S + r)로 저장하고 r = (S + r) − Q(S + r)을 다음 청크로 넘긴다(`quant.Intervention.prefill_image`). n_vis ≤ 256이면 청크가 1개라 `8ef` = `8`이다. 분할 forward 자체의 정확성: fp32에서 2단 분할 vs 단일 forward의 마지막 행 logits 최대 차이 5.9e-5 (CPU), 6.3e-5 (MPS), logit 크기 약 21. bf16에서는 같은 비교가 0.25로, 분할에 따른 bf16 반올림 차이다. 모든 조건이 같은 2단 경로를 쓰므로 조건 간 비교에는 영향이 없지만, stock `model(**inputs)` 단일 forward 수치와는 bf16에서 정확히 같지 않다.

## 추가: 0.8B 구조 (budget.py 입력)

| 항목 | 값 |
|---|---|
| 층 수 | 24 |
| 배치 | `[GDN, GDN, GDN, Attn] × 6` (`full_attention_interval = 4`), Attn 층 인덱스 3, 7, 11, 15, 19, 23 |
| GDN 층 / Attn 층 | 18 / 6 |
| state shape (층당) | `[B, 16, 128, 128]` (linear_num_value_heads 16, d_k 128, d_v 128), fp32 |
| conv state (층당) | `[B, 6144, 4]`, bf16, 개입하지 않음 |
| KV head 수 | 2 (`num_key_value_heads`), Q head 8 |
| head dim | 256 |
| KV 원소/토큰 | 2 × 6 × 2 × 256 = 6,144 |
| state 원소(고정) | 18 × 16 × 128 × 128 = 4,718,592 |
| vision | depth 12, patch 16, spatial_merge 2, temporal_patch 2 → 시각 토큰 1개 = 32×32 px |

예: 448×448 이미지(n_vis 196) + 텍스트 22토큰, full: KV 2.68 MB(16bit), state 18.87 MB(32bit).

## 개입 정의에서 내린 구현 선택 (확인 사항 아님, 설계)

- scope=image의 stage-1 계산은 깨끗하다. 이미지 마지막 토큰 직후 캐시에 남은 값(시각 토큰 KV 행, 모든 GDN state)을 한 번 바꾸고, 질문 prefill과 decode는 바뀐 메모리를 읽는다.
- conv1d state(마지막 4개 입력, 즉 시각 토큰 3개 + 현재)는 hybridquant 관례대로 건드리지 않는다. 따라서 state zero에서도 마지막 시각 토큰 몇 개의 국소 정보가 conv state로 질문에 전달될 수 있다. 논문에서 이 경로를 명시해야 한다.
- KV zero는 K와 V를 0으로 둔다. 이 행은 attention에서 제거되지 않고 점수 0(가중치 exp(0))을 받으며 값 0을 기여한다.
- noise는 hybridquant `QuantSpec(noise_match=True)`: 같은 텐서에 RTN을 한 번 수행해 텐서 전체 오차 RMS를 재고, 그 RMS의 가우시안을 더한다. 시드 0 고정.
- 양자화 그룹: 마지막 차원 128 (KV는 head_dim 256 → 토큰·헤드당 2그룹, state는 d_v 128 → 행당 1그룹), 비대칭 RTN.

## 데이터

| task | HF id | revision | split (shard) | 원 표본 수 | subset | 카드 라이선스 |
|---|---|---|---|---|---|---|
| DocVQA | `lmms-lab/DocVQA` (config DocVQA) | 539088ef8a8a | validation (6 shards) | 5,349 | 200 | apache-2.0 (HF 카드 표기). 원 DocVQA 이용 약관은 UNVERIFIED |
| ChartQA | `lmms-lab/ChartQA` | 9e63b7df1592 | test (1 shard) | 2,500 | 200 | HF 카드에 표기 없음. 원 데이터셋 라이선스 UNVERIFIED |
| TextVQA | `lmms-lab/textvqa` | 9c0699cd1976 | validation (3 shards) | 5,000 | 200 | HF 카드에 표기 없음. 원 데이터셋 라이선스 UNVERIFIED |
| POPE | `lmms-lab/POPE` (COCO val2014 이미지) | 4db1276663df | test (3 shards; random/popular/adversarial 혼합) | 9,000 | 300 | HF 카드에 표기 없음. COCO 이미지 약관은 UNVERIFIED |

- subset: `scripts/make_subsets.py`, 시드 0, `numpy.random.default_rng(0).choice(N, n, replace=False)` 후 인덱스 정렬. `configs/subsets/{task}.json`에 sample_id, 전역 인덱스, shard 파일, shard 내 행 번호를 저장. 로더는 행을 읽을 때 id가 일치하는지 검사한다.
- ChartQA는 id 열이 없어 `chartqa-{전역 인덱스}`를 id로 쓴다.
- 프롬프트(`data.PROMPTS`, 네 태스크 공통, lmms-eval 방식): `"{question}\nAnswer the question using a single word or phrase."`. system 메시지 없음. 템플릿은 `processor.apply_chat_template(add_generation_prompt=True)`.
- gold(teacher-forced NLL용 한 문자열): DocVQA `answers[0]`, ChartQA `answer`, TextVQA 10개 중 최빈값(동률이면 먼저 나온 것), POPE `answer`(소문자 yes/no 그대로). 채점은 전체 정답 목록 사용.
- 채점: DocVQA ANLS(τ=0.5), ChartQA relaxed accuracy(수치 5%), TextVQA VQA accuracy(10개 leave-one-out 평균), POPE yes/no 일치(표본) 및 데이터셋 수준 F1. 정규화는 `metrics.normalize` 한 곳(소문자, 소수점 외 마침표·구두점 제거, 관사 제거, 공백 정리). 표준 ANLS는 관사를 지우지 않으므로 공식 수치와 약간 다를 수 있다.
- 이미지 해상도 상한: `max_pixels = 1,310,720` (= 시각 토큰 1280개). configs의 `max_pixels`로 고정. 결정 근거는 decision_log D-4.

<!-- L1:start -->
## L1 결과

`scripts/check_l1.py`, 모델 Qwen/Qwen3.5-0.8B, device=mps, commit 7324e17434, 총 5.9s. 값은 기록만 하고 해석하지 않는다.

- (a) input_ids=(1, 218) vis_span=(4, 200) n_vis=196 n_text=22 image_grid_thw=[[1, 28, 28]] pixel_values=(784, 1536)
- (b) unhooked two-stage gold NLL/token = 0.052222; stock single-pass model(**inputs) = 0.056059 (|diff| 3.84e-03, bf16 split vs one pass)
- (c) hooked full NLL/token = 0.052222, |(c)-(b)| = 0.00e+00 (PASS at 1e-4); state_calls=216 kv_calls=72
- (d) one prefill forward, state writes: n_vis=0 -> 18, n_vis=196 -> 18, n_vis=1024 -> 18; GDN layers=18. Whole-cell totals (with decode steps) are in the table: blind 306, image 216

| run | n_vis | gold NLL/token | pred | score | state_calls | kv_calls | n_chunks | sec |
|---|---|---|---|---|---|---|---|---|
| full (hooks on, pass-through) | 196 | 0.0522 | 'TWO MEMORIES 2026' | 1.00 | 216 | 72 | 1 | 0.53 |
| blind | 0 | 4.3306 | "I don't see an image in your message. Could you please upload or describe" | 0.00 | 306 | 102 | 0 | 0.47 |
| kv zero | 196 | 3.3459 | 'The text written in the image is:\n\n**“The text is written in' | 0.00 | 324 | 108 | 1 | 0.68 |
| state zero | 196 | 0.0470 | 'TWO MEMORIES 2026' | 1.00 | 216 | 72 | 1 | 0.52 |
| kv4 / state8ef | 196 | 0.0671 | 'TWO MEMORIES 2026' | 1.00 | 234 | 78 | 1 | 0.56 |
| full, 1024px image | 1024 | 3.7295 | 'MINOR' | 0.00 | 90 | 30 | 1 | 1.15 |
| state 8ef, 1024px image | 1024 | 3.7506 | 'MINOR' | 0.00 | 162 | 54 | 4 | 1.23 |

<!-- L1:end -->
