# DLA HW1 — ASR

DEMO: https://colab.research.google.com/drive/1aJfdXH5l7wICcO35dowNGvGof7yxhuKn?usp=sharing

## 1. Лучшая модель

DeepSpeech2 + CTC: 2 свертки, 5 двунаправленных GRU по 512, FC 512. Всего 25,3 млн параметров.
Обучение с нуля на `train-other-500`: 50 эпох, AdamW, warmup и cosine LR.
Аугментации: шум, скорость, громкость и Time masking. Лучший checkpoint — эпоха 44 по WER на `dev-other`

Corpus-метрики на полном `test-other`:

| Декодирование | CER | WER |
| --- | --- | --- |
| Argmax | 10,45% | 27,16% |
| Beam search, beam = 5, без LM | 10,38% | 27,01% |

## 2. Подготовка

Для обучения использовался Python 3.10

```bash
git clone https://github.com/teplov-andrew/DLA_HW1.git
cd DLA_HW1
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p saved
```

Скачайте [веса](https://drive.google.com/file/d/1VXPSjwfcp4giV9rwo5deb8ArhZT8qb6y/view) в `saved/ds2_final.pth`

[demo_samples](https://drive.google.com/drive/folders/1HwYuBE3ROgpkUFOO0Ze6zWor_tY24_rN?usp=sharing)

```
Мой диск/
└── DLA_HW1/
    ├── ds2_final.pth
    └── demo_samples/
        ├── audio/
        │   ├── 121-127105-0034.wav
        │   ├── 4970-29093-0015.wav
        │   ├── 6938-70848-0027.wav
        │   ├── 8280-266249-0031.wav
        │   └── 8555-292519-0011.wav
        ├── transcriptions/
        │   ├── 121-127105-0034.txt
        │   ├── 4970-29093-0015.txt
        │   ├── 6938-70848-0027.txt
        │   ├── 8280-266249-0031.txt
        │   └── 8555-292519-0011.txt
```

Зависимости для Colab: [requirements-colab.txt](requirements-colab.txt)

## 3. Обучение

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u train.py -cn deepspeech2 \
  datasets.train.part=train-other-500 transforms=deepspeech2 \
  model.rnn_hidden=512 model.rnn_layers=5 model.fc_hidden=512 \
  optimizer=adamw optimizer.lr=0.0003 optimizer.weight_decay=0.01 \
  lr_scheduler=cosine_warmup lr_scheduler.warmup_steps=1000 \
  lr_scheduler.start_factor=0.01 lr_scheduler.min_lr=0.000001 \
  dataloader.batch_size=64 dataloader.num_workers=8 dataloader.pin_memory=false \
  trainer.device=cuda trainer.n_epochs=50 trainer.resume_from=null \
  +trainer.skip_oom=false 'trainer.monitor="min dev_other_WER_(Argmax)"' \
  writer.mode=offline writer.run_name=ds2_final_reproduce
```

LibriSpeech скачивается автоматически. Веса сохраняются в `saved/ds2_final_reproduce/`, логи — в `.cometml-runs/`

## 4. Инференс

Положите записи в `data/custom/audio/`, эталоны — в `data/custom/transcriptions/`
Имена должны совпадать: `sample.wav` и `sample.txt`

```bash
python inference.py data_dir=data/custom \
  inferencer.from_pretrained=saved/ds2_final.pth \
  inferencer.decoder=beam_search inferencer.beam_size=5 \
  inferencer.save_path=predictions

python calc_metrics.py reference_dir=data/custom/transcriptions \
  prediction_dir=predictions/custom output=metrics.json
```

Для argmax замените `inferencer.decoder=beam_search` на `inferencer.decoder=greedy`

## 5. Ссылки

- [Отчет в Comet](https://www.comet.com/andrew-teplov/hw1-asr/reports/LTelprGWGxMuiP7C1AtKX6W6s)
- [Финальное обучение](https://www.comet.com/andrew-teplov/hw1-asr/lk2w72sohhrrtugo9a6gift2dh81zll3)
- [Демонстрация аугментаций](https://www.comet.com/andrew-teplov/hw1-asr/78b00351fc3940fbb309a62dbe546681)
- [Веса модели](https://drive.google.com/file/d/1VXPSjwfcp4giV9rwo5deb8ArhZT8qb6y/view)
