"""
Веб‑интерфейс для AISingersCoverGen.

Данный модуль реализует веб‑приложение на основе Gradio 4.x. Интерфейс
позволяет загружать или записывать аудио, выбирать модель RVC и
настраивать параметры конвертации: изменение высоты (Transpose),
форманта, метод извлечения F0, коэффициенты RMS, фильтр, защиту
согласных, AutoTune и формат вывода. Синие и зелёные оттенки
используются в качестве основной цветовой схемы.

Для запуска сервера из командной строки:

    python webui.py --share

Параметр `--share` включает публичную ссылку, полезную для работы в
Google Colab.
"""

from __future__ import annotations

import argparse
import os
import logging

import gradio as gr

from .rvc import RVCPipeline

logger = logging.getLogger(__name__)


def create_interface() -> gr.Blocks:
    """Создать интерфейс Gradio и вернуть объект Blocks."""
    pipeline = RVCPipeline(models_dir="rvc_models")

    # Загрузить список моделей для выпадающего списка
    models = pipeline.list_models()
    model_names = list(models.keys()) or ["Нет моделей"]

    # Функция для обработки аудио
    def process(
        model_name: str,
        input_audio: tuple[int, bytes] | None,
        transpose: float,
        formant: float,
        f0_method: str,
        index_rate: float,
        filter_radius: int,
        rms_mix_rate: float,
        protect: float,
        autotune: bool,
        output_format: str,
    ) -> gr.Audio:
        """Обработать аудио и вернуть файл для скачивания."""
        if input_audio is None:
            raise gr.Error("Пожалуйста, загрузите файл или запишите голос.")
        # input_audio приходит в формате (sample_rate, data) для записанных
        # аудио либо (filepath, None) для загруженных файлов.
        if isinstance(input_audio[1], (bytes, bytearray)):
            # для аудио, записанного в браузере
            sr = input_audio[0]
            import io
            import soundfile as sf
            data, _ = sf.read(io.BytesIO(input_audio[1]))
            audio = data.astype(float)
        else:
            # для загруженного файла — первый элемент tuple содержит путь
            filepath = input_audio[0]
            audio, sr = pipeline.load_audio(filepath)
        # Обработать аудио
        converted, sr = pipeline.infer_voice(
            model_name=model_name,
            audio=audio,
            sr=sr,
            pitch_change=transpose,
            formant=formant,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            autotune=autotune,
        )
        # Сохранить результат во временный файл
        import tempfile
        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, f"output.{output_format}")
        pipeline.save_audio(converted, sr, filename, format=output_format)
        return gr.Audio.update(value=filename, label="Результат")

    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="teal",
        neutral_hue="slate",
    )) as demo:
        gr.Markdown(
            """
            # AISingersCoverGen

            Создавайте AI‑кавера с помощью Retrieval‑based Voice Conversion (RVC).
            Загрузите или запишите свой голос, выберите модель и настройте
            параметры, чтобы получить уникальный результат. Реализована
            поддержка современных методов извлечения F0, включая FCPE и
            гибридные варианты.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    label="Модель",
                    value=model_names[0],
                    interactive=True,
                )
                f0_method = gr.Dropdown(
                    choices=[
                        "pm",
                        "harvest",
                        "dio",
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                        "rmvpe+fcpe",
                    ],
                    value="rmvpe",
                    label="Метод F0",
                    interactive=True,
                    info="Выберите алгоритм извлечения высоты тона",
                )
                transpose = gr.Slider(
                    minimum=-16,
                    maximum=16,
                    step=0.1,
                    value=0.0,
                    label="Transpose (полутона)",
                    interactive=True,
                )
                formant = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    value=0.0,
                    label="Форманта",
                    interactive=True,
                )
                index_rate = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Index Rate",
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=7,
                    step=1,
                    value=3,
                    label="Filter radius",
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.25,
                    label="RMS mix",
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    step=0.01,
                    value=0.33,
                    label="Protect",
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    value=False,
                    label="AutoTune",
                    interactive=True,
                )
                output_format = gr.Radio(
                    choices=["wav", "mp3"], value="wav", label="Формат вывода"
                )
                submit_btn = gr.Button("Сгенерировать")
            with gr.Column(scale=1):
                input_audio = gr.Audio(
                    source="upload", label="Входной голос", type="filepath",
                    interactive=True,
                    show_download_button=False,
                )
                record_audio = gr.Audio(
                    source="microphone", label="Записать голос", type="numpy",
                    interactive=True,
                    show_download_button=False,
                )
                # Показываем результат
                output_audio = gr.Audio(
                    label="Результат", type="filepath", interactive=False
                )
        # Приоритет для записи: если audio из микрофона присутствует, используем его
        def choose_input(input_file, record_file):
            if record_file is not None and record_file[1] is not None:
                return record_file
            return input_file

        submit_btn.click(
            fn=process,
            inputs=[
                model_dropdown,
                gr.State(choose_input(input_audio, record_audio)),
                transpose,
                formant,
                f0_method,
                index_rate,
                filter_radius,
                rms_mix_rate,
                protect,
                autotune,
                output_format,
            ],
            outputs=output_audio,
        )
        return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск веб‑интерфейса AISingersCoverGen")
    parser.add_argument("--share", action="store_true", help="Разрешить публичную ссылку (полезно в Colab)")
    parser.add_argument("--listen", action="store_true", help="Слушать на всех интерфейсах (0.0.0.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    demo = create_interface()
    # Определяем хост и флаг share
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    demo.launch(share=args.share, server_name=server_name)


if __name__ == "__main__":
    main()