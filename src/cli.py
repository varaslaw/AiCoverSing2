"""
Командный интерфейс для AISingersCoverGen.

С помощью этого скрипта вы можете конвертировать аудио через RVC
без запуска веб‑интерфейса. Все параметры аналогичны тем, что
доступны в Gradio UI. В качестве результата создаётся файл
`output.{format}` в текущем каталоге, если имя файла не указано явно.
"""

from __future__ import annotations

import argparse
import os
import logging

from .rvc import RVCPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI для AISingersCoverGen")
    parser.add_argument("-i", "--input", required=True, help="Путь к исходному аудио")
    parser.add_argument("-m", "--model", required=True, help="Имя модели (папка в rvc_models)")
    parser.add_argument("-o", "--output", default=None, help="Путь к выходному файлу")
    parser.add_argument("--pitch_change", type=float, default=0.0, help="Изменение высоты тона (полутона, дробные значения)")
    parser.add_argument("--formant", type=float, default=0.0, help="Изменение форманты")
    parser.add_argument("--f0_method", type=str, default="rmvpe", help="Метод извлечения F0 (pm, harvest, dio, crepe, crepe-tiny, rmvpe, fcpe, rmvpe+fcpe)")
    parser.add_argument("--index_rate", type=float, default=0.5, help="Доля индекса модели (0–1)")
    parser.add_argument("--filter_radius", type=int, default=3, help="Радиус медианного фильтра")
    parser.add_argument("--rms_mix_rate", type=float, default=0.25, help="Баланс RMS (0–1)")
    parser.add_argument("--protect", type=float, default=0.33, help="Степень защиты согласных (0–0.5)")
    parser.add_argument("--autotune", action="store_true", help="Включить AutoTune")
    parser.add_argument("--output_format", choices=["wav", "mp3"], default="wav", help="Формат выхода")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = RVCPipeline(models_dir="rvc_models")
    # Загрузить аудио
    audio, sr = pipeline.load_audio(args.input)
    # Конвертировать
    converted, sr = pipeline.infer_voice(
        model_name=args.model,
        audio=audio,
        sr=sr,
        pitch_change=args.pitch_change,
        formant=args.formant,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        autotune=args.autotune,
    )
    # Определить имя файла
    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.input))
        args.output = f"{base}_converted.{args.output_format}"
    # Сохранить
    pipeline.save_audio(converted, sr, args.output, format=args.output_format)
    print(f"Готово! Сохранён файл: {args.output}")


if __name__ == "__main__":
    main()