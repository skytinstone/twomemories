# 실행 기록 (experiment log)

수치의 출처는 `results/**/*.jsonl`과 `results/**/compute_log.jsonl`이다. 이 문서는 무엇을 언제 어떤 커밋으로 돌렸는지만 적는다.

## 2026-11-07 로컬 (MacBook Pro M3 Pro, MPS, 유닛 0)

| 단계 | 명령 | 커밋 | 결과 |
|---|---|---|---|
| C1 | `pytest -q` | 4c7387e 이후 작업 트리 | 14 passed, 6.35s |
| C2 | `python scripts/check_l1.py` | (L1 절 참조) | docs/assumptions.md "L1 결과" |
| C3 | `python scripts/make_subsets.py` | — | configs/subsets/*.json (한 번만 실행) |
| C3 | `python scripts/run_phase.py --phase preflight --limit 5 --out results/local` | 8db117f79e | results/local/, 24셀 120줄, 스키마 통과, 4분 22초. 재실행 시 전부 skip(resume) |
| C3 | `python analysis/make_tables.py --model Qwen/Qwen3.5-0.8B --allow-local --results results/local` | 8db117f79e | Table 2/5 생성, 미평가 셀 n/e |
| C3 | `python scripts/check_l2.py` | 8db117f79e | greedy 5/5 동일, batch2 최대 차이 7.2e-3 → batch 1 고정(D-13) |

로컬 수치는 경로 검증용이며 논문에 쓰지 않는다.

## 2026-11-07 Colab preflight 1차 (v0.1-preflight, 중단)

| 단계 | 결과 |
|---|---|
| 셀 2 setup | 완료. 리소스 패널 약 8.88 유닛/시간, GPU RAM 80 GB |
| 셀 4 preflight | 4B DocVQA blind 셀 5샘플 완료(0.46 s/sample, VRAM 피크 8.7 GB) 후 KVfull_Sfull 셀에서 Drive 캐시 parquet 손상으로 중단(D-18) |
| 셀 5 요약 | dry-run 루프 `{ph}` 미확장(D-20). 사용 유닛 약 3.2(설치·다운로드 포함) |

수정 후 `v0.1.1-preflight`로 재실행. blind DocVQA 5줄은 cell_hash가 같아 resume으로 건너뛴다(그 줄의 commit은 188a6dd).

## 2026-11-07 맥 작업 사고 기록

run_all.sh 이어하기 시뮬레이션 중 임시 clone이 실패한 채 명령이 이어져, 실제 레포에서 테스트용 sed와 run_all이 실행됐다. 가짜 결과 6줄(`results/m/...`)이 원격 브랜치 `results/0432a52`로 push됐다. 확인 후 브랜치를 삭제하고 run_all.sh를 되돌렸다. main과 태그는 영향 없음. 이후 시뮬레이션은 `set -e` 서브셸, 로컬 bare 원격, `GH_TOKEN` 제거로 다시 했다.

## 2026-11-07 Colab preflight 2차 (v0.1.1-preflight, 완료)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 8.88 유닛/시간(리소스 패널) |
| 셀 | compute_log 91셀(기대 92). 빠진 1셀은 1차 실행의 DocVQA blind 기록으로 추정. 1차의 `_tag_phase`가 Drive 위에서 compute_log를 다시 쓰다 잃은 것으로 보이나 UNVERIFIED. 결과 jsonl은 덧붙이기만 하므로 영향 없음 |
| 속도 | 1.47 s/sample(행 기준 평균). blind 0.87, EF 1.66–1.99, 토큰 삭제 1.34–1.37 |
| VRAM 피크 | 9.3 GB |
| 유닛 | preflight 1.64 |

상한은 D-25.

## 2026-11-07 로컬 phase 5 점검 (0.8B, MPS, 검증용, 논문 미사용)

`python scripts/run_phase.py --phase phase5_channels --model Qwen/Qwen3.5-0.8B --limit 5 --out results/local`: 12셀 신규, full·blind 8셀 skip. 5샘플 retained(참고만): KV drop ≈ KV zero(DocVQA 0.14 대 0.13, TextVQA 0.31 대 0.32), state zeroconv는 state zero와 비슷하거나 약간 높음, drop+zeroconv는 DocVQA −0.10, TextVQA 0.02(0 근처). ChartQA는 CI가 너무 넓어 판단 불가.

## 2026-11-07~08 Colab 본 실험 v0.2-main (phase 1–4 완료) — **전부 무효**

| 항목 | 값 |
|---|---|
| 범위 | 4B phase 1–3, 9B phase 4. 결과는 원격 브랜치 `results/v0.2-main`(2aae6af)에 보관, main에는 병합하지 않음 |
| 유닛(compute_log, 행 seconds 기준) | preflight 1.6, phase1 36.4, phase2 20.0, phase3 14.4, phase4 18.8, 합 91.3. 실제 청구량은 리소스 패널로 확인 필요(UNVERIFIED) |
| 증상 | 모든 셀(full 포함) 정확도 0.000. pred가 전부 "The user wants to …" 같은 생각 문장. gold NLL 토큰 평균 7.2–24.9(로컬 0.8B full은 0.002–0.9) |
| 원인 | Qwen3.5 4B/9B chat template은 `enable_thinking`이 없으면 thinking이 기본이라 생성 프롬프트가 `<think>\n`으로 끝난다. 0.8B template은 반대로 non-thinking이 기본(`<think>\n\n</think>\n\n`). `_encode`가 `enable_thinking`을 넘기지 않아 4B/9B는 열린 생각 블록 바로 뒤에서 정답 NLL을 재고 decode했다. 4B/9B 프로세서로 프롬프트 끝을 디코드해 확인 |
| 놓친 이유 | 로컬 검증은 0.8B만 썼고, Colab preflight 2차 점검은 속도·VRAM·셀 수만 봤다(정확도 미확인). run_all이 phase를 자동으로 이어 돌아 phase별 수동 확인(CLAUDE.md)도 없었다 |
| 조치 | D-29 |

v0.3-channels(phase 5)도 같은 버그가 있으므로 돌리지 않는다.

## 2026-11-09 맥 0.8B 본 실험 완료와 두 단계 확인

- phase 1, 2, 3, 5 (`results/mac`): 104셀 전부 완료(태스크당 25조건 + blind, 200/200/200/300샘플). 로그 4개 모두 exit=0, `-dirty` 줄 0. phase 2–5 약 2.5시간(23:50–02:20).
- 두 단계 vs 단일 패스(`scripts/check_twostage.py`, 태스크당 처음 50샘플, `results/checks/twostage_0.8B.jsonl`): 평균 NLL/토큰 차이(single − two-stage) DocVQA −0.0040, ChartQA −0.0013, TextVQA +0.0071, POPE −0.0070. blind–full 차이 대비 0.07–0.21%. 샘플별 최대 |차이| 0.15–0.32(bf16 분할 반올림). D-32의 UNVERIFIED 해소.
- Colab core4b(D-33): 5샘플 점검 통과(full > blind 4태스크, 짧은 답, 0.63 s/샘플, 9.3 GB, 0.22유닛). 100샘플 실행 중.

## 2026-11-09 Colab 4B 확인(core4b) 완료
- 결과 브랜치 `results/v0.5-core4b`(ece01ee). 이 브랜치는 맥 결과 이전 커밋(4e78f13)에서 갈라져 통째로 합치면 `results/mac`이 지워지므로, 4B 파일과 compute_log만 `results/colab/`으로 가져왔다.
- 28셀 × 100샘플, 중복 0, commit 전부 4e78f13(-dirty 없음), GPU A100-SXM4-40GB, 0.58 s/샘플, 추정 4.01유닛(5샘플 점검 포함), 최대 9.2 GB. transformers 5.22.1, torch 2.14.1+cu130.
- 정확도 full/blind: DocVQA 0.92/0.06, ChartQA 0.80/0.11, TextVQA 0.79/0.06, POPE 0.88/0.59.
- 정확도 retained: KV zero 0.05/0.06/0.06/0.41, state zero 1.04/1.00/1.02/1.07, KV4·S8·KV4+S8ef ≥ 0.99. 0.8B보다 state가 덜 필요하다(ChartQA 0.66 → 1.00). POPE 대소문자 효과 재현: 원본 대문자 0%, state zero 후 68%.
- 표: `paper/tables/table_4b_vs_08b.tex`(생성), `paper/tables/Qwen3.5-4B/`. 논문 §6.4와 남은 TBD를 채움.
