"""
Скрипт для скачивания необходимых моделей.

Этот модуль предназначен для загрузки предобученных моделей RVC и
сопутствующих файлов. Поскольку в среде выполнения (например, в Google
Colab) не всегда доступны внешние сети, данный скрипт предоставляет
пример того, как можно скачивать модели с помощью библиотеки
``mega.py`` и других источников. Вы можете изменить список ссылок
под свои нужды.

При запуске без аргументов будут загружены базовые модели: MDXNET для
разделения вокала, модель ``hubert`` для извлечения аудио‑фич и
модель ``rmvpe`` для F0. Загрузка FCPE‑модели требует отдельного
шагa, так как её веса размещены на HuggingFace.
"""

from __future__ import annotations

import os
import logging

from mega import Mega  # type: ignore
from tenacity import retry, wait_fixed, stop_after_attempt

logger = logging.getLogger(__name__)


@retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
def download_from_url(url: str, dest: str) -> None:
    """Скачать файл из произвольной ссылки с повторными попытками.

    Библиотека ``mega.py`` используется для MEGA‑ссылок. Для обычных HTTP/HTTPS
    используется requests. Если загрузка прерывается, будет предпринято
    несколько попыток.
    """
    if url.startswith("https://mega.nz"):
        m = Mega()
        m.login()  # анонимная сессия
        logger.info("MEGA-загрузка: %s", url)
        m.download_url(url, dest)
    else:
        import requests
        logger.info("Загрузка: %s", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    logger.info("Файл сохранён: %s", dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Каталоги для моделей
    base_dir = Path(__file__).resolve().parent.parent
    mdxnet_models_dir = base_dir / "mdxnet_models"
    rvc_models_dir = base_dir / "rvc_models"
    mdxnet_models_dir.mkdir(exist_ok=True)
    rvc_models_dir.mkdir(exist_ok=True)

    # Ссылки на модели MDXNET (vocal/instrument separation)
    # Используются модели из репозитория TRvlvr/model_repo (UVR-MDX-Net).
    mdx_models = {
        "UVR-MDX-NET-Voc_FT.onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
        "UVR_MDXNET_KARA_2.onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
        "Reverb_HQ_By_FoxJoy.onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx",
    }

    # Базовые модели RVC (hubert и rmvpe) скачиваются с HuggingFace
    rvc_models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    }

    # Скачать MDXNET модели
    for filename, url in mdx_models.items():
        dest = mdxnet_models_dir / filename
        if dest.exists():
            logger.info("Файл уже существует, пропускаем: %s", filename)
            continue
        try:
            download_from_url(url, str(dest))
        except Exception as e:
            logger.error("Не удалось скачать %s: %s", filename, e)

    # Скачать RVC модели (hubert_base, rmvpe)
    for filename, url in rvc_models.items():
        dest = rvc_models_dir / filename
        if dest.exists():
            logger.info("Файл уже существует, пропускаем: %s", filename)
            continue
        try:
            download_from_url(url, str(dest))
        except Exception as e:
            logger.error("Не удалось скачать %s: %s", filename, e)

    # Модель FCPE поставляется вместе с библиотекой torchfcpe.
    logger.info(
        "Загрузка завершена. Модель FCPE поставляется через pip (torchfcpe)."
    )


if __name__ == "__main__":
    from pathlib import Path
    main()