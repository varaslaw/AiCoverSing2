"""
Модуль rvc содержит высокоуровневый класс `RVCPipeline` для загрузки
моделей RVC, подготовки аудио и вызова алгоритмов конвертации голоса.

В рамках упрощённого примера этот класс демонстрирует структуру
функций, необходимых для работы с RVC‑моделями. Реальная загрузка
весов и инференс модели оставлены за пределами данного репозитория,
поскольку они зависят от конкретных предобученных данных. Пользователь
может расширить этот класс, подключив свои модели.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

from .vc_infer_pipeline import VC, change_rms

logger = logging.getLogger(__name__)


class RVCPipeline:
    """Высокоуровневый пайплайн для работы с RVC‑моделями.

    Этот класс управляет загрузкой моделей, подготовкой аудио и
    вызовом конвертации. В данной реализации модель не выполняет
    реальной конверсии голоса; вместо этого исходный аудиосигнал
    возвращается без изменений. Вы можете дополнить метод
    `infer_voice` своими алгоритмами конвертации.
    """

    def __init__(self, models_dir: str = "rvc_models",
                 device: Optional[str] = None) -> None:
        self.models_dir = models_dir
        self.device = device
        # В этом словаре будут храниться загруженные модели
        self.models: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Работа с моделями
    def list_models(self) -> Dict[str, str]:
        """Вернуть словарь доступных моделей в каталоге models_dir.

        Каждая модель хранится в отдельной папке и должна содержать
        файлы `.pth` и `.index`. Функция возвращает карту из имени
        модели в путь к её каталогу.
        """
        models = {}
        if not os.path.isdir(self.models_dir):
            logger.warning(
                "Каталог моделей %s не найден. Создайте его и добавьте RVC‑модели.",
                self.models_dir,
            )
            return models
        for name in sorted(os.listdir(self.models_dir)):
            path = os.path.join(self.models_dir, name)
            if os.path.isdir(path):
                pth_files = [f for f in os.listdir(path) if f.endswith(".pth")]
                index_files = [f for f in os.listdir(path) if f.endswith(".index")]
                if pth_files and index_files:
                    models[name] = path
        return models

    def load_model(self, model_name: str) -> None:
        """Загрузить модель по имени. Пока выполняется заглушка.

        В реальной реализации здесь следует загрузить веса из файлов
        `.pth` и `.index` и проинициализировать модель. После загрузки
        модель сохраняется в self.models.
        """
        if model_name in self.models:
            return
        models = self.list_models()
        if model_name not in models:
            raise ValueError(f"Модель {model_name} не найдена в {self.models_dir}")
        model_dir = models[model_name]
        # Здесь должна быть логика загрузки модели.
        # Пока просто сохраняем путь в словарь.
        self.models[model_name] = {
            "dir": model_dir,
            "pth": [f for f in os.listdir(model_dir) if f.endswith(".pth")][0],
            "index": [f for f in os.listdir(model_dir) if f.endswith(".index")][0],
        }
        logger.info("Загружена модель %s", model_name)

    # ------------------------------------------------------------------
    # Предобработка аудио
    def load_audio(self, filepath: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
        """Загрузить аудио файл и преобразовать в моно с заданной частотой.

        Возвращает кортеж (аудиоданные, частота дискретизации).
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        y, orig_sr = librosa.load(filepath, sr=sr, mono=True)
        return y.astype(np.float32), sr

    # ------------------------------------------------------------------
    # Инференс модели (заглушка)
    def infer_voice(
        self,
        model_name: str,
        audio: np.ndarray,
        sr: int,
        pitch_change: float = 0.0,
        formant: float = 0.0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.5,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        autotune: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Преобразовать голос с использованием выбранной модели и параметров.

        В демонстрационной реализации аудиоданные возвращаются без изменений.
        Вы можете интегрировать свой алгоритм конверсии, используя модуль
        vc_infer_pipeline и другие классы.
        """
        # Убедимся, что модель загружена
        self.load_model(model_name)
        # Здесь могла бы быть логика вызова VC-пайплайна. Например:
        #   vc = VC(tgt_sr=sr, config=...)
        #   f0 = vc.get_f0(audio, ...)
        #   converted_audio = vc.vc(...)
        # Для упрощения просто копируем аудио
        logger.debug(
            "Вызван infer_voice: model=%s, pitch_change=%s, f0_method=%s", 
            model_name, pitch_change, f0_method,
        )
        return audio, sr

    # ------------------------------------------------------------------
    # Сохранение результатов
    def save_audio(
        self, audio: np.ndarray, sr: int, filepath: str, format: str = "wav"
    ) -> None:
        """Сохранить аудио в файл. Поддерживаются WAV и MP3 (если установлен ffmpeg).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if format.lower() == "wav":
            sf.write(filepath, audio, sr)
        elif format.lower() == "mp3":
            # Для MP3 используем ffmpeg через soundfile
            import ffmpeg
            # Сохраняем во временный WAV
            tmp_wav = filepath + ".tmp.wav"
            sf.write(tmp_wav, audio, sr)
            (
                ffmpeg.input(tmp_wav)
                .output(filepath, format="mp3", audio_bitrate="192k")
                .run(overwrite_output=True, quiet=True)
            )
            os.remove(tmp_wav)
        else:
            raise ValueError(f"Неизвестный формат: {format}")
