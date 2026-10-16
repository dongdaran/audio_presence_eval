# AudioSet Presence Evaluation

## 준비

- Vast.ai GPU: A100 80GB
- 입력 파일: `caption_positive_samples.tsv`, `caption_negative_samples.tsv`, `ontology.json` (`data/`)
- 실행 설정: `config/experiment.yaml`
- YouTube 오디오 다운로드를 위해 네트워크와 `ffmpeg`가 필요합니다.
- 키 파일을 준비합니다: `cp .env.example .env` 후 필요한 키를 입력합니다.

## 실행

저장소 루트에서 이미지를 빌드합니다.

```bash
docker build -f experiments/audio_presence_eval/Dockerfile \
  -t audioset-presence-eval experiments/audio_presence_eval
```

아래 명령은 모델별 LLM 평가 후 CLAP, CNN14 점수를 계산하고 요약을 만듭니다. 결과는 `results/`에 저장됩니다.

```bash
set -e
run_eval() {
  docker run --rm --gpus all \
    --env-file .env \
    -v "$PWD/experiments/audio_presence_eval/data:/workspace/experiments/audio_presence_eval/data:ro" \
    -v "$PWD/experiments/audio_presence_eval/model_cache:/workspace/.cache/huggingface" \
    -v "$PWD/experiments/audio_presence_eval/audio_cache:/workspace/experiments/audio_presence_eval/audio_cache" \
    -v "$PWD/experiments/audio_presence_eval/results:/workspace/experiments/audio_presence_eval/results" \
    audioset-presence-eval "$@"
}

for model in gemini_3_pro kimi_audio_7b audio_flamingo_3 qwen3_omni_instruct; do
  run_eval --task llm --model "$model"
done

for task in clap cnn14 report; do
  run_eval --task "$task"
done
```

결과 요약: `experiments/audio_presence_eval/results/README.md`

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
