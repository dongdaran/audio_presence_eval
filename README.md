# AudioSet Presence Evaluation

## StemFoley test 평가

`config/stemfoley.yaml`은 `data/stemfoley_contam_v1/test_labels.csv`의 476개
로컬 오디오를 사용합니다. 정답은 `source_present` (`target_audible`의 yes/no),
난이도는 `difficulty` (clean → easy, contaminated → hard)입니다.
undetermined는 제외합니다. 원본 데이터와 기존 실험 결과는 유지됩니다.

기존 설치 절차로 `.venv`와 `.venv-kimi`를 준비한 뒤 실행합니다.

```bash
# 모델마다 2개씩 먼저 확인
bash scripts/run_stemfoley.sh --limit 2
# Qwen → Audio Flamingo → Kimi → CLAPScore → CNN14 (매핑된 74개) → report
bash scripts/run_stemfoley.sh
```

개별 실행:

```bash
.venv/bin/python scripts/run_experiment.py --config config/stemfoley.yaml --task llm --model qwen3_omni_instruct
.venv/bin/python scripts/run_experiment.py --config config/stemfoley.yaml --task llm --model audio_flamingo_3
.venv-kimi/bin/python scripts/run_experiment.py --config config/stemfoley.yaml --task llm --model kimi_audio_7b
.venv/bin/python scripts/run_experiment.py --config config/stemfoley.yaml --task clap
```

결과는 `results/stemfoley/`에 저장하며 성공한 행은 재실행 시 건너뜁니다.
TSV에 정답과 난이도를 함께 기록하고 README에 easy/hard × yes/no별 성능을 집계합니다.
`--limit` 출력은 `smoke_` 파일로 저장하며 전체 평가 보고서에는 포함하지 않습니다.
CLAP은 target 문장과의 유사도이며 자동 임계값을 적용하지 않습니다.
CNN14는 target이 AudioSet 클래스명과 정확히 일치하는 74개만 평가합니다.
나머지 402개는 건너뛰며, 보고서의 CNN14 성능은 이 부분집합에 대한 점수입니다.

## 준비

- Vast.ai GPU: A100 80GB
- 입력 파일: `caption_positive_samples.tsv`, `caption_negative_samples.tsv`, `ontology.json` (`data/`)
- 실행 설정: `config/experiment.yaml`
- YouTube 오디오 다운로드를 위해 네트워크와 `ffmpeg`가 필요합니다.
- 키 파일을 준비합니다: `cp .env.example .env` 후 필요한 키를 입력합니다.

## Vast.ai에서 Docker 없이 실행 (uv)

프로젝트 폴더(`/workspace/audio_presence_eval`)에서 실행합니다.

```bash
uv venv --python /venv/main/bin/python .venv
uv pip install --python .venv/bin/python -c constraints-gpu.txt \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -c constraints-gpu.txt -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

CUDA 12.6용 PyTorch를 사용합니다. 현재 Vast.ai 드라이버에서 CUDA 13용 PyTorch는 GPU를 초기화하지 못합니다.

`.env.example`을 참고해 프로젝트 루트의 `.env`에 아래 값을 입력합니다. 프로그램이 자동으로 읽으므로 `export`가 필요 없습니다. `.env`는 Git과 Docker 빌드 컨텍스트에서 제외됩니다.

```dotenv
OPENROUTER_API_KEY=본인의_OpenRouter_API_키
# 선택: 오디오 입력을 지원하는 다른 OpenRouter 모델
# OPENROUTER_MODEL=google/gemini-3.1-pro-preview
```

`gemini_3_pro`는 기존 결과 파일 이름과의 호환을 위한 설정 키이며, 실제 API 호출은 OpenRouter의 `google/gemini-3.1-pro-preview`를 사용합니다. 실제 모델 ID는 결과 TSV에 기록됩니다. Kimi, Audio Flamingo, Qwen은 로컬 GPU 모델입니다.

먼저 소량 실행으로 API와 오디오 다운로드를 확인합니다.

```bash
.venv/bin/python scripts/run_experiment.py --task llm --model gemini_3_pro --limit 2
```

Gemini Argon도 같은 `.env`의 `OPENROUTER_API_KEY`를 사용합니다. `OPENROUTER_MODEL`의 영향을 받지 않으며, 실제 모델 ID는 `google/gemini-4-argon`입니다.

```bash
.venv/bin/python scripts/run_experiment.py --task llm --model gemini_argon --limit 2
```

전체 평가:

```bash
set -e
for model in gemini_3_pro gemini_argon kimi_audio_7b audio_flamingo_3 qwen3_omni_instruct; do
  if [ "$model" = kimi_audio_7b ]; then
    .venv-kimi/bin/python scripts/run_experiment.py --task llm --model "$model"
  else
    .venv/bin/python scripts/run_experiment.py --task llm --model "$model"
  fi
done
for task in clap cnn14 report; do
  .venv/bin/python scripts/run_experiment.py --task "$task"
done
```

OpenRouter 오디오 요청은 [공식 오디오 입력 규격](https://openrouter.ai/docs/guides/overview/multimodal/audio)을 따릅니다.

Kimi-Audio는 Transformers 4.51.3를 사용하는 전용 `.venv-kimi`에서 실행합니다. 다른 모델의 `.venv`는 Transformers 5 환경을 유지합니다. 설치 스크립트는 현재 Vast.ai의 Linux x86_64 / Python 3.12 / CUDA 12.6 환경용입니다.

```bash
bash scripts/install_kimi.sh
.venv-kimi/bin/python scripts/run_experiment.py --task llm --model kimi_audio_7b --limit 2
```

설치 스크립트는 공식 Kimi 소스와 서브모듈, PyTorch 2.7.1 및 이에 맞는 공식 FlashAttention 휠을 설치하고 GPU 연산을 검사합니다. 모델 가중치는 첫 실행 때 다운로드됩니다. `.env`는 이 환경에서도 자동으로 읽습니다.

결과와 요약은 `results/`에 저장됩니다. `--limit` 실행 결과에는 `smoke_` 접두사가 붙습니다.

샘플 결과 검증 (모델별 2개 행 실행 후):

```bash
.venv/bin/python scripts/check_smoke_results.py --limit 2
```

`results/validation/README.md`에 모델별 성공 건수와 응답/점수를 기록합니다. 결과 행이 없거나 오류·빈 응답·유효하지 않은 점수가 있으면 종료 코드 1을 반환합니다. OpenRouter HTTP 402는 계정 크레딧 부족이며 `.env`의 키에 연결된 계정 잔액을 확인해야 합니다.

Kimi의 메타데이터 없는 Whisper safetensors 로딩은 Transformers 4.51.3에서 검증했습니다. Transformers 모델은 오디오를 모델 dtype으로 변환하고, 텍스트에만 패딩 옵션을 적용해 모델별 오디오 패딩 규칙을 유지합니다. CLAP는 Hugging Face 캐시에서 체크포인트를 읽습니다. CNN14 미러는 Zenodo 원본 체크섬과 일치하는지 확인한 뒤 사용합니다.


## 결과 예시

아래 숫자는 출력 형식을 보여주는 예시이며 실제 결과가 아닙니다. 결과 요약에는 LLM 분류 성능과 CLAP/CNN14의 정답 `yes`·`no`별 평균 점수가 기록됩니다.

LLM 성능 예시:

| Model | N scored | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) |
|---|---:|---:|---:|---:|---:|
| qwen3_omni_instruct | 25,000 | 0.82 | 0.80 | 0.85 | 0.82 |

CLAPScore / CNN14 평균 점수 예시:

| Score | Reference label | N | Pooled mean | Pooled SD | Per-MID macro mean |
|---|---|---:|---:|---:|---:|
| CLAPScore | yes | 11,740 | 0.42 | 0.12 | 0.42 |
| CLAPScore | no | 13,250 | 0.31 | 0.11 | 0.31 |
| CNN14 | yes | 11,600 | 0.18 | 0.15 | 0.18 |
| CNN14 | no | 13,000 | 0.09 | 0.10 | 0.09 |

점수표는 yes/no별 전체 평균과 클래스별 점수를 함께 제공합니다. `Pooled mean`은 샘플 전체의 평균, `Pooled SD`는 샘플 점수의 표준편차, `Per-MID macro mean`은 MID별 평균을 구한 뒤 클래스마다 같은 비중으로 다시 평균한 값입니다. 클래스별 표에는 MID마다 yes/no 샘플 수와 평균, `Yes minus no` 차이가 나옵니다. 점수 차이가 크면 그 클래스의 yes와 no 샘플이 모델 점수상 더 잘 구분된다는 뜻입니다.

원본 예측과 점수는 각각 `llm_<model>.tsv`, `clap_scores.tsv`, `cnn14_scores.tsv`에 저장됩니다.
