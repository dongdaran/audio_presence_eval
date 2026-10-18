# Qwen2.5-Omni gain 평가 결과

476개 음원 × 7개 gain = 3,332건 완료, 실패 0건. 정답: yes 349개, no 127개.

| Gain (dB) | Accuracy | Macro F1 | 0dB 대비 판정 변경 | 오답→정답 | 정답→오답 |
|---:|---:|---:|---:|---:|---:|
| -30 | 68.28% | 0.6678 | 54 | 16 | 38 |
| -20 | 70.59% | 0.6905 | 41 | 15 | 26 |
| -10 | 72.48% | 0.7077 | 30 | 14 | 16 |
| -5 | 71.85% | 0.7007 | 19 | 7 | 12 |
| 0 | 72.90% | 0.7103 | 0 | 0 | 0 |
| 5 | 72.06% | 0.7020 | 12 | 4 | 8 |
| 10 | 72.90% | 0.7103 | 18 | 9 | 9 |

0dB와 +10dB의 accuracy 및 macro F1은 같지만, 18개 음원의 판정은 다름. +10dB에서 9개가 개선되고 9개가 악화됨.

입력 파형에 10 ** (gain_db / 20)을 곱하며 clipping하지 않음. +5dB에서 170개, +10dB에서 297개 음원이 절댓값 1을 넘음.

실행: RTX 3090 24GB, batch 64 OOM 후 32로 축소. 최대 PyTorch allocated 20.07 GiB, reserved 21.73 GiB.

## Yes/no 도출

프롬프트: Does this audio contain the sound of {caption}? Respond using only one word: yes or no.

오디오와 질문을 입력한 forward 결과의 마지막 입력 위치 logits[:, -1]을 사용함. 이는 첫 답변 토큰에 대한 vocabulary 전체의 raw logits임.

yes/Yes/YES 및 앞 공백 변형을 토큰화한 모든 token ID를 중복 제거해 yes 집합을 만들고 no/No/NO도 동일하게 처리함. 여러 subword로 분리되면 그 ID들도 집합에 포함됨.

yes_logit = logsumexp(logits[yes_token_ids])
no_logit = logsumexp(logits[no_token_ids])
logit_margin = yes_logit - no_logit

margin > 0이면 yes, 그 외에는 no. 동점은 no. 생성 문자열을 파싱하는 방식이 아닌 두 token 집합 사이의 점수 비교이며, AAD/TAD 보정이나 silent 오디오 분기는 사용하지 않음.

logit은 확률이 아님. 두 집합에만 정규화한 yes 상대 확률은 sigmoid(logit_margin)이며 전체 vocabulary 확률 또는 보정된 정답 확률을 의미하지 않음.

실제 yes token IDs: [7414, 9454, 9693, 9834, 14004, 14080]
실제 no token IDs: [902, 2152, 2308, 2753, 5664, 8996]
