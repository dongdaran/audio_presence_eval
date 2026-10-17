# StemFoley 대상 소리 존재 여부 평가 결과

전체 평가 대상: **476**개. 정답은 target_audible이며, 판정 불가(undetermined) 항목은 제외합니다.

## LLM 소리 존재 여부 예측

`yes`는 대상 소리가 있음, `no`는 없음입니다. 정확도는 정답/평가 개수, 정밀도는 yes 정답/yes 예측 개수, 재현율은 yes 정답/실제 yes 개수를 표시하며 괄호 안은 퍼센트입니다. 무효 응답과 추론 실패는 별도 집계하고 지표 분모에서 제외합니다. F1은 정밀도와 재현율을 종합한 0~1 점수로 표시합니다. TP·FN·FP·TN 열은 정답→예측 순서이며 단위는 개입니다.

| 모델 | 유효 평가 개수 | 무효/실패 개수 | 정확도 (정답/평가) | 정밀도 (yes) | 재현율 (yes) | F1 (yes) | TP (있음→있음) | FN (있음→없음) | FP (없음→있음) | TN (없음→없음) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_flamingo_3 | 476 | 0 | 145/476 (30.46%) | 48/78 (61.54%) | 48/349 (13.75%) | 0.2248 | 48 | 301 | 30 | 97 |
| kimi_audio_7b | 476 | 0 | 369/476 (77.52%) | 273/304 (89.80%) | 273/349 (78.22%) | 0.8361 | 273 | 76 | 31 | 96 |
| qwen3_omni_instruct | 476 | 0 | 381/476 (80.04%) | 284/314 (90.45%) | 284/349 (81.38%) | 0.8567 | 284 | 65 | 30 | 97 |

## 설정된 모델 ID

- `gemini_3_pro`: `google/gemini-3.1-pro-preview` (최대 생성 토큰: `4096`)
- `gemini_argon`: `google/gemini-4-argon` (최대 생성 토큰: `4096`)
- `kimi_audio_7b`: `moonshotai/Kimi-Audio-7B-Instruct` (최대 생성 토큰: `1`)
- `audio_flamingo_3`: `nvidia/audio-flamingo-3-hf` (최대 생성 토큰: `1`)
- `qwen3_omni_instruct`: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (최대 생성 토큰: `1`)

- 최대 생성 토큰: `1`
- 점수 모델 배치 크기: `16`
- LLM Transformers 마이크로 배치 크기: `16`
- 설정된 GPU: `NVIDIA A100 80GB`

## 정답 레이블별 CLAPScore / CNN14 평균 점수

CLAPScore는 오디오와 문장 임베딩 사이의 코사인 유사도입니다. CNN14는 매핑된 AudioSet 클래스 ID(MID)의 점수입니다. 두 점수 모두 임계값을 적용한 yes/no 판정이나 정확도가 아닙니다.

전체 평균과 표준편차는 유효 점수가 있는 샘플을 기준으로 계산합니다. 소리별 평균의 평균은 원래 대상 소리 이름별 평균에 동일한 가중치를 줍니다. 클래스별 집계도 MID가 아닌 원래 대상 소리 이름을 기준으로 합니다. 아래 클래스별 표는 실제 소리가 있는 경우와 없는 경우의 평균 및 차이를 보여줍니다. 대상 소리 이름은 원본 데이터의 영문 표기를 유지합니다. NA는 계산 가능한 값이 없다는 뜻입니다.

| 점수 종류 | 정답 (yes=있음, no=없음) | 유효 점수 개수 | 소리 종류 수 | 전체 평균 | 전체 표준편차 | 소리별 평균의 평균 |
|---|---|---:|---:|---:|---:|---:|
| clap | yes | 349 | 148 | 0.305593 | 0.157569 | 0.294014 |
| clap | no | 127 | 76 | 0.112434 | 0.123745 | 0.110492 |
| cnn14 | yes | 60 | 21 | 0.458060 | 0.289802 | 0.447365 |
| cnn14 | no | 11 | 7 | 0.001844 | 0.002323 | 0.002350 |

## CLAPScore 클래스별 점수

| 클래스 ID (MID) | 대상 소리 (원문) | 있음 개수 | 있음 평균 | 없음 개수 | 없음 평균 | 있음 평균 − 없음 평균 |
|---|---|---:|---:|---:|---:|---:|
| 미매핑 | Acoustic guitar strumming | 2 | 0.402858 | 0 | NA | NA |
| 미매핑 | Acrobat impacts | 0 | NA | 3 | 0.225093 | NA |
| 미매핑 | Adult vocalization | 0 | NA | 3 | -0.062338 | NA |
| 미매핑 | amplified vocalization | 0 | NA | 2 | 0.114798 | NA |
| /m/028ght | applause | 0 | NA | 3 | 0.121354 | NA |
| 미매핑 | Arena crowd ambience | 3 | 0.205844 | 0 | NA | NA |
| 미매핑 | audience applause | 2 | 0.574293 | 0 | NA | NA |
| /m/018w8 | Basketball bounce | 3 | 0.448282 | 0 | NA | NA |
| 미매핑 | Basketball bounces | 6 | 0.453696 | 0 | NA | NA |
| 미매핑 | Basketball dribbles | 0 | NA | 2 | 0.199311 | NA |
| /m/018vs | bass guitar | 3 | 0.238847 | 0 | NA | NA |
| /m/01c3q | Bassoon | 3 | 0.214728 | 0 | NA | NA |
| 미매핑 | BMX bicycle rolling | 1 | 0.210285 | 1 | -0.042727 | 0.253012 |
| 미매핑 | Body and grass rustle | 0 | NA | 1 | -0.010374 | NA |
| 미매핑 | Book and page handling | 0 | NA | 1 | 0.201402 | NA |
| 미매핑 | Bowed strings | 4 | 0.253856 | 1 | 0.120342 | 0.133514 |
| 미매핑 | Bowl-piece handling | 2 | 0.264852 | 0 | NA | NA |
| 미매핑 | Bowling alley ambience | 2 | 0.315184 | 1 | 0.265176 | 0.050007 |
| 미매핑 | Bowling ball rolling | 2 | 0.405763 | 1 | 0.229328 | 0.176434 |
| 미매핑 | Brown-suited man vocalizing | 3 | 0.263194 | 0 | NA | NA |
| 미매핑 | Car engine | 2 | 0.310291 | 0 | NA | NA |
| 미매핑 | Carriage ambience | 3 | -0.054728 | 0 | NA | NA |
| 미매핑 | Cattle lowing | 2 | 0.389332 | 0 | NA | NA |
| /m/01xqw | cello | 2 | 0.237402 | 0 | NA | NA |
| 미매핑 | Child crying | 3 | 0.353919 | 0 | NA | NA |
| 미매핑 | Child vocalization | 2 | 0.253596 | 0 | NA | NA |
| 미매핑 | Child voice | 2 | 0.435999 | 0 | NA | NA |
| 미매핑 | Close pedestrian pass-bys | 0 | NA | 1 | 0.035249 | NA |
| 미매핑 | clothing rustle | 1 | 0.082080 | 1 | -0.012057 | 0.094137 |
| 미매핑 | Court shoe scuffs | 1 | 0.243185 | 2 | 0.226822 | 0.016363 |
| 미매핑 | Crib and bedding handling | 1 | 0.249645 | 1 | 0.041140 | 0.208505 |
| 미매핑 | Crowd ambience | 1 | -0.025731 | 2 | 0.012704 | -0.038435 |
| 미매핑 | crowd ambience | 3 | 0.348127 | 0 | NA | NA |
| 미매핑 | crowd cheering | 1 | 0.253351 | 2 | 0.123582 | 0.129769 |
| 미매핑 | crowd noise | 3 | 0.289056 | 0 | NA | NA |
| 미매핑 | crowd rustle | 3 | 0.127840 | 0 | NA | NA |
| 미매핑 | Cupped-hand whistle | 2 | 0.505248 | 0 | NA | NA |
| 미매핑 | Cymbal strikes | 3 | 0.240047 | 0 | NA | NA |
| 미매핑 | Dark-haired man vocalizing | 2 | 0.171046 | 1 | 0.089847 | 0.081199 |
| 미매핑 | Dog snuffling or licking | 1 | 0.263983 | 2 | -0.029689 | 0.293672 |
| /m/02hnl | Drum kit | 2 | 0.392484 | 1 | 0.093911 | 0.298573 |
| /m/02hnl | drum kit | 5 | 0.379713 | 1 | 0.162049 | 0.217663 |
| 미매핑 | Drum-kit playing | 0 | NA | 1 | 0.129075 | NA |
| 미매핑 | Electric bass | 3 | 0.361441 | 0 | NA | NA |
| /m/02sgy | Electric guitar | 1 | 0.058800 | 1 | 0.144315 | -0.085516 |
| /m/02sgy | electric guitar | 5 | 0.299235 | 0 | NA | NA |
| 미매핑 | Electric guitar playing | 3 | 0.491436 | 0 | NA | NA |
| 미매핑 | Electronic keyboard | 2 | -0.009682 | 0 | NA | NA |
| 미매핑 | electronic keyboard | 2 | 0.037331 | 0 | NA | NA |
| /t/dd00004 | Female singing | 1 | 0.468460 | 2 | -0.014773 | 0.483233 |
| 미매핑 | female singing voice | 2 | 0.446148 | 0 | NA | NA |
| 미매핑 | female vocal | 2 | 0.348114 | 1 | 0.056237 | 0.291876 |
| 미매핑 | Female vocalization | 0 | NA | 1 | -0.028685 | NA |
| 미매핑 | Fingernails tapping page | 2 | 0.281602 | 0 | NA | NA |
| 미매핑 | Fire engine engine and road noise | 2 | 0.374739 | 0 | NA | NA |
| 미매핑 | Fire engine siren | 1 | 0.215738 | 1 | 0.196057 | 0.019680 |
| 미매핑 | flag flutter | 0 | NA | 1 | -0.146132 | NA |
| 미매핑 | Floor and body contacts | 1 | 0.446775 | 2 | 0.450573 | -0.003798 |
| 미매핑 | Flower lei rustling | 0 | NA | 3 | -0.089773 | NA |
| /m/0l14j_ | Flute | 2 | 0.400569 | 0 | NA | NA |
| 미매핑 | Foreground trombone | 2 | 0.431762 | 1 | 0.196735 | 0.235028 |
| 미매핑 | foreground violin | 3 | 0.395394 | 0 | NA | NA |
| 미매핑 | Four men vocalizing | 3 | 0.233813 | 0 | NA | NA |
| 미매핑 | group singing | 2 | 0.368276 | 0 | NA | NA |
| 미매핑 | group vocalization | 3 | 0.475410 | 0 | NA | NA |
| 미매핑 | Gym ambience | 2 | 0.397202 | 0 | NA | NA |
| 미매핑 | Hand claps | 1 | 0.187378 | 2 | 0.147389 | 0.039988 |
| 미매핑 | hand claps | 0 | NA | 3 | 0.133535 | NA |
| 미매핑 | Hand strikes on wood | 3 | 0.430410 | 0 | NA | NA |
| 미매핑 | Hand-to-body slaps | 2 | 0.514794 | 0 | NA | NA |
| 미매핑 | Harmonium | 3 | 0.235654 | 0 | NA | NA |
| 미매핑 | harmonium | 3 | 0.271019 | 0 | NA | NA |
| 미매핑 | Hoof impacts | 0 | NA | 3 | 0.243892 | NA |
| 미매핑 | Horse hoofbeats | 2 | 0.205872 | 0 | NA | NA |
| 미매핑 | Horse landing | 3 | 0.354530 | 0 | NA | NA |
| 미매핑 | Indoor hall ambience | 2 | 0.357818 | 0 | NA | NA |
| 미매핑 | Indoor room tone | 2 | 0.146425 | 0 | NA | NA |
| 미매핑 | Infant vocalization | 6 | 0.284990 | 0 | NA | NA |
| 미매핑 | Keyboard | 5 | 0.194771 | 3 | 0.043572 | 0.151198 |
| 미매핑 | large drum | 0 | NA | 3 | 0.069005 | NA |
| 미매핑 | Left tabla | 2 | 0.378080 | 1 | 0.136239 | 0.241840 |
| 미매핑 | left ukulele | 2 | 0.485539 | 0 | NA | NA |
| 미매핑 | male speech | 2 | 0.031604 | 0 | NA | NA |
| 미매핑 | Male vocalization | 2 | 0.082505 | 3 | 0.130388 | -0.047883 |
| 미매핑 | Male voice | 2 | 0.130067 | 0 | NA | NA |
| 미매핑 | male voice | 4 | 0.072405 | 0 | NA | NA |
| 미매핑 | Mallet strikes | 3 | 0.377517 | 0 | NA | NA |
| 미매핑 | Man in gray overshirt speaking | 1 | 0.247815 | 1 | 0.228761 | 0.019054 |
| 미매핑 | Man in olive jacket speaking | 1 | 0.340829 | 2 | 0.104638 | 0.236191 |
| 미매핑 | Man in striped sweater speaking | 1 | 0.293794 | 2 | 0.169404 | 0.124390 |
| 미매핑 | Man's vocalization | 2 | 0.074496 | 1 | 0.054739 | 0.019757 |
| 미매핑 | Man's voice | 2 | 0.240780 | 0 | NA | NA |
| 미매핑 | men vocalizing | 1 | 0.117315 | 2 | 0.057959 | 0.059356 |
| 미매핑 | Microphone vocalization | 0 | NA | 3 | 0.256248 | NA |
| 미매핑 | Microphone-side men's voices | 2 | 0.202135 | 0 | NA | NA |
| 미매핑 | motorcycle engine | 2 | 0.480827 | 0 | NA | NA |
| 미매핑 | Moving Jeep | 3 | 0.302010 | 0 | NA | NA |
| 미매핑 | nasal applicator actuation | 1 | 0.123807 | 1 | 0.102669 | 0.021138 |
| /m/05pd6 | orchestra | 2 | 0.485894 | 0 | NA | NA |
| 미매핑 | Outdoor ambience | 2 | 0.221140 | 0 | NA | NA |
| 미매핑 | Outdoor arena ambience | 2 | 0.342519 | 1 | 0.204265 | 0.138254 |
| 미매핑 | Outdoor court ambience | 2 | 0.201654 | 1 | 0.071455 | 0.130199 |
| 미매핑 | Outdoor plaza ambience | 1 | -0.057878 | 2 | -0.011755 | -0.046123 |
| 미매핑 | Passing auto-rickshaw | 0 | NA | 2 | 0.036652 | NA |
| 미매핑 | passing freight train | 3 | 0.244201 | 0 | NA | NA |
| 미매핑 | Passing motorcycle | 0 | NA | 1 | 0.093951 | NA |
| /m/05r5c | piano | 3 | 0.235106 | 0 | NA | NA |
| 미매핑 | piano playing | 2 | 0.472305 | 0 | NA | NA |
| 미매핑 | Pins struck | 1 | 0.443373 | 2 | 0.118437 | 0.324936 |
| 미매핑 | Player exertion sounds | 2 | 0.034933 | 0 | NA | NA |
| 미매핑 | Player footfalls | 1 | 0.332650 | 2 | 0.393936 | -0.061285 |
| 미매핑 | Player footsteps | 0 | NA | 2 | 0.112706 | NA |
| 미매핑 | Plaza crowd ambience | 3 | 0.018888 | 0 | NA | NA |
| 미매핑 | Presenter speech | 2 | 0.196943 | 0 | NA | NA |
| 미매핑 | race start signal | 0 | NA | 2 | 0.112748 | NA |
| 미매핑 | Racket and shuttle impacts | 1 | 0.233984 | 2 | 0.160133 | 0.073850 |
| 미매핑 | Rear brass section | 3 | 0.307512 | 0 | NA | NA |
| 미매핑 | rear cello | 2 | 0.257968 | 1 | 0.207663 | 0.050305 |
| 미매핑 | rigging clatter | 3 | 0.125849 | 0 | NA | NA |
| 미매핑 | Right tabla | 2 | 0.411937 | 1 | 0.136350 | 0.275588 |
| 미매핑 | right ukulele | 2 | 0.521963 | 0 | NA | NA |
| 미매핑 | right violin | 3 | 0.315770 | 0 | NA | NA |
| 미매핑 | Rim and net | 1 | 0.320969 | 2 | 0.354262 | -0.033293 |
| 미매핑 | road traffic | 2 | 0.438374 | 0 | NA | NA |
| 미매핑 | Roadside traffic ambience | 1 | 0.095824 | 2 | 0.050109 | 0.045716 |
| 미매핑 | runner footfalls | 0 | NA | 2 | 0.059427 | NA |
| 미매핑 | Runners' footsteps on stones | 2 | 0.289015 | 0 | NA | NA |
| 미매핑 | running footsteps | 0 | NA | 3 | -0.030681 | NA |
| 미매핑 | Seated man speaking | 1 | 0.293110 | 1 | 0.321933 | -0.028824 |
| 미매핑 | sedan driving | 3 | 0.216345 | 0 | NA | NA |
| 미매핑 | Shoe squeaks | 3 | 0.108818 | 0 | NA | NA |
| 미매핑 | Shoe-floor contact | 1 | -0.050467 | 2 | 0.155498 | -0.205965 |
| 미매핑 | Singing voice | 5 | 0.384072 | 0 | NA | NA |
| 미매핑 | singing voice | 4 | 0.236383 | 2 | 0.142152 | 0.094231 |
| 미매핑 | Skateboard grinding ledge | 3 | 0.431728 | 0 | NA | NA |
| 미매핑 | Skateboard landing | 3 | 0.444314 | 0 | NA | NA |
| 미매핑 | Skateboard leaving ledge | 1 | 0.430242 | 2 | 0.195045 | 0.235196 |
| 미매핑 | Skateboard rolling | 2 | 0.346892 | 1 | 0.034768 | 0.312124 |
| 미매핑 | Small guitar strumming | 3 | 0.288844 | 0 | NA | NA |
| /m/06rvn | Snare drum | 3 | 0.275788 | 0 | NA | NA |
| /m/09x0r | speech | 2 | 0.097830 | 0 | NA | NA |
| 미매핑 | Stage men's dialogue | 1 | 0.040271 | 2 | -0.005002 | 0.045272 |
| 미매핑 | stick drum | 1 | 0.068114 | 1 | 0.098186 | -0.030071 |
| 미매핑 | street fireworks | 3 | 0.403517 | 0 | NA | NA |
| 미매핑 | Surrounding road traffic | 2 | 0.522048 | 0 | NA | NA |
| 미매핑 | SUV driving | 2 | 0.349988 | 0 | NA | NA |
| /m/01p970 | Tabla | 4 | 0.338368 | 0 | NA | NA |
| /m/01p970 | tabla | 2 | 0.419882 | 0 | NA | NA |
| 미매핑 | tabla drumming | 3 | 0.491300 | 0 | NA | NA |
| 미매핑 | tanpura | 3 | 0.286859 | 0 | NA | NA |
| /m/0h9mv | tire squeal | 3 | 0.613031 | 0 | NA | NA |
| 미매핑 | Tire-road friction | 2 | 0.281463 | 0 | NA | NA |
| 미매핑 | tissue rustling | 0 | NA | 1 | -0.027241 | NA |
| 미매핑 | Tom drums | 2 | 0.256146 | 1 | 0.088370 | 0.167776 |
| /m/0284vy3 | train horn | 2 | 0.695283 | 0 | NA | NA |
| /m/07c6l | trombone | 3 | 0.564996 | 1 | 0.104773 | 0.460223 |
| 미매핑 | Trowel contacting bin contents | 0 | NA | 2 | 0.164006 | NA |
| /m/07gql | Trumpet | 4 | 0.532279 | 0 | NA | NA |
| /m/07gql | trumpet | 5 | 0.474460 | 2 | 0.169491 | 0.304969 |
| 미매핑 | vehicle engine | 3 | 0.224737 | 0 | NA | NA |
| 미매핑 | vehicle horn | 2 | 0.539290 | 0 | NA | NA |
| 미매핑 | vehicle movement | 3 | 0.312837 | 0 | NA | NA |
| 미매핑 | violin | 3 | 0.319459 | 0 | NA | NA |
| 미매핑 | Walking footsteps | 0 | NA | 2 | 0.030634 | NA |
| 미매핑 | water against hull | 3 | 0.172125 | 0 | NA | NA |
| /m/03m9d0z | wind | 3 | 0.513007 | 0 | NA | NA |
| 미매핑 | Wind over hillside | 2 | 0.398993 | 0 | NA | NA |
| 미매핑 | Woman singing | 3 | 0.401247 | 0 | NA | NA |
| 미매핑 | Woman speaking | 4 | 0.272859 | 0 | NA | NA |
| 미매핑 | woman speaking | 2 | 0.267519 | 0 | NA | NA |
| 미매핑 | woman vocalization | 0 | NA | 1 | -0.077266 | NA |
| 미매핑 | Woman vocalizing | 2 | 0.183182 | 1 | 0.055357 | 0.127825 |
| 미매핑 | woman's voice | 2 | 0.122965 | 0 | NA | NA |
| 미매핑 | Yangqin | 3 | 0.067769 | 0 | NA | NA |

## CNN14 클래스별 점수

| 클래스 ID (MID) | 대상 소리 (원문) | 있음 개수 | 있음 평균 | 없음 개수 | 없음 평균 | 있음 평균 − 없음 평균 |
|---|---|---:|---:|---:|---:|---:|
| /m/028ght | applause | 0 | NA | 3 | 0.000954 | NA |
| /m/018w8 | Basketball bounce | 3 | 0.193360 | 0 | NA | NA |
| /m/018vs | bass guitar | 3 | 0.201485 | 0 | NA | NA |
| /m/01c3q | Bassoon | 0 | NA | 0 | NA | NA |
| /m/01xqw | cello | 2 | 0.244280 | 0 | NA | NA |
| /m/02hnl | Drum kit | 2 | 0.200378 | 1 | 0.000493 | 0.199885 |
| /m/02hnl | drum kit | 5 | 0.620796 | 1 | 0.005527 | 0.615269 |
| /m/02sgy | Electric guitar | 1 | 0.303645 | 1 | 0.007392 | 0.296252 |
| /m/02sgy | electric guitar | 5 | 0.483397 | 0 | NA | NA |
| /t/dd00004 | Female singing | 1 | 0.612816 | 2 | 0.001255 | 0.611561 |
| /m/0l14j_ | Flute | 2 | 0.873668 | 0 | NA | NA |
| /m/05pd6 | orchestra | 2 | 0.275311 | 0 | NA | NA |
| /m/05r5c | piano | 3 | 0.230589 | 0 | NA | NA |
| /m/06rvn | Snare drum | 3 | 0.553595 | 0 | NA | NA |
| /m/09x0r | speech | 2 | 0.829265 | 0 | NA | NA |
| /m/01p970 | Tabla | 4 | 0.836480 | 0 | NA | NA |
| /m/01p970 | tabla | 2 | 0.594780 | 0 | NA | NA |
| /m/0h9mv | tire squeal | 3 | 0.102853 | 0 | NA | NA |
| /m/0284vy3 | train horn | 2 | 0.422907 | 0 | NA | NA |
| /m/07c6l | trombone | 3 | 0.547034 | 1 | 0.000164 | 0.546870 |
| /m/07gql | Trumpet | 4 | 0.820537 | 0 | NA | NA |
| /m/07gql | trumpet | 5 | 0.353875 | 2 | 0.000667 | 0.353208 |
| /m/03m9d0z | wind | 3 | 0.093606 | 0 | NA | NA |

## 참고 사항

- 다운로드·추론 실패 및 CNN14 클래스 미매핑 항목은 결과 TSV에 남아 있으며, 유효 점수가 없으면 평균에서 제외합니다.
- CLAPScore는 유사도 점수이며, 보정된 확률이 아닙니다.


## 난이도·정답별 상세 결과

LLM은 정답/유효 평가 개수 (정확도)를 표시합니다. CLAPScore/CNN14는 임계값 없이 계산한 평균 점수이며 퍼센트가 아닙니다. 실패·무효·미매핑 항목은 지표에서 제외하며, 저장된 개수와 유효 평가 개수를 함께 표시합니다. easy는 쉬움(오염 없음), hard는 어려움(오염 있음)입니다.

| 모델 | 난이도 | 정답 (yes=있음, no=없음) | 저장된 개수 | 유효 평가 개수 | 정답/평가 (정확도) 또는 평균 점수 |
|---|---|---|---:|---:|---:|
| llm_audio_flamingo_3 | easy | yes | 220 | 220 | 31/220 (14.09%) |
| llm_audio_flamingo_3 | easy | no | 14 | 14 | 10/14 (71.43%) |
| llm_audio_flamingo_3 | hard | yes | 129 | 129 | 17/129 (13.18%) |
| llm_audio_flamingo_3 | hard | no | 113 | 113 | 87/113 (76.99%) |
| llm_kimi_audio_7b | easy | yes | 220 | 220 | 175/220 (79.55%) |
| llm_kimi_audio_7b | easy | no | 14 | 14 | 9/14 (64.29%) |
| llm_kimi_audio_7b | hard | yes | 129 | 129 | 98/129 (75.97%) |
| llm_kimi_audio_7b | hard | no | 113 | 113 | 87/113 (76.99%) |
| llm_qwen3_omni_instruct | easy | yes | 220 | 220 | 189/220 (85.91%) |
| llm_qwen3_omni_instruct | easy | no | 14 | 14 | 10/14 (71.43%) |
| llm_qwen3_omni_instruct | hard | yes | 129 | 129 | 95/129 (73.64%) |
| llm_qwen3_omni_instruct | hard | no | 113 | 113 | 87/113 (76.99%) |
| clap_scores | easy | yes | 220 | 220 | 0.339358 |
| clap_scores | easy | no | 14 | 14 | 0.167997 |
| clap_scores | hard | yes | 129 | 129 | 0.248010 |
| clap_scores | hard | no | 113 | 113 | 0.105550 |
| cnn14_scores | easy | yes | 48 | 46 | 0.508823 |
| cnn14_scores | easy | no | 0 | 0 | NA |
| cnn14_scores | hard | yes | 15 | 14 | 0.291268 |
| cnn14_scores | hard | no | 11 | 11 | 0.001844 |

## 난이도별 합산 결과 (yes/no 합계)

easy = 쉬움(오염 없음), hard = 어려움(오염 있음). LLM은 정답/유효 평가 개수 (정확도)를 표시합니다. 실패·무효 응답은 정확도에서 제외합니다. CLAPScore/CNN14는 유효 점수의 평균이며 정확도나 퍼센트가 아닙니다. CNN14는 매핑 가능한 대상만 평가합니다.

| 모델 | 난이도 | 저장된 개수 | 유효 평가 개수 | 실패/무효/미매핑 개수 | 정답/유효 평가 (정확도) 또는 평균 점수 |
|---|---|---:|---:|---:|---:|
| audio_flamingo_3 | easy | 234 | 234 | 0 | 41/234 (17.52%) |
| audio_flamingo_3 | hard | 242 | 242 | 0 | 104/242 (42.98%) |
| kimi_audio_7b | easy | 234 | 234 | 0 | 184/234 (78.63%) |
| kimi_audio_7b | hard | 242 | 242 | 0 | 185/242 (76.45%) |
| qwen3_omni_instruct | easy | 234 | 234 | 0 | 199/234 (85.04%) |
| qwen3_omni_instruct | hard | 242 | 242 | 0 | 182/242 (75.21%) |
| CLAPScore | easy | 234 | 234 | 0 | 0.329106 |
| CLAPScore | hard | 242 | 242 | 0 | 0.181490 |
| CNN14 | easy | 48 | 46 | 2 | 0.508823 |
| CNN14 | hard | 26 | 25 | 1 | 0.163922 |
