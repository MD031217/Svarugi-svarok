# import warnings
# from io import BytesIO
#
# from fastapi import FastAPI, HTTPException, Request
# from PIL import Image, ImageOps, UnidentifiedImageError
# from starlette.concurrency import run_in_threadpool
#
# from app.classes import DATASET_CLASSES
# from app.config import Settings
# from app.detector import Detector, ModelUnavailableError
# from app.schemas import DatasetClass, HealthResponse, PredictionResponse
#
#
# def decode_image(content: bytes, max_pixels: int) -> Image.Image:
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("error", Image.DecompressionBombWarning)
#             with Image.open(BytesIO(content)) as source:
#                 if source.format not in {"JPEG", "PNG"}:
#                     raise HTTPException(415, "Only JPEG and PNG images are supported")
#                 if source.width * source.height > max_pixels:
#                     raise HTTPException(413, "Image exceeds the pixel limit")
#                 return ImageOps.exif_transpose(source).convert("RGB")
#     except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
#         raise HTTPException(400, "Invalid or corrupted image") from exc
#     except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
#         raise HTTPException(413, "Image exceeds the pixel limit") from exc
#
#
# def create_app(settings: Settings | None = None) -> FastAPI:
#     config = settings if settings is not None else Settings()
#     api = FastAPI(title=config.app_name, debug=config.debug, version="0.1.0")
#     detector = Detector(config)
#
#     @api.get("/health", response_model=HealthResponse, tags=["health"])
#     def health() -> HealthResponse:
#         """API liveness; does not imply trained weights are available."""
#         return HealthResponse()
#
#     @api.get("/api/v1/classes", response_model=list[DatasetClass], tags=["train"])
#     def classes() -> tuple[DatasetClass, ...]:
#         """Source train class IDs, independent of loaded model weights."""
#         return DATASET_CLASSES
#
#     @api.post(
#         "/api/v1/predictions",
#         response_model=PredictionResponse,
#         tags=["detection"],
#         responses={
#             400: {"description": "Invalid image"},
#             413: {"description": "Image is too large"},
#             415: {"description": "Unsupported image format"},
#             503: {"description": "Detection model unavailable"},
#         },
#         openapi_extra={
#             "requestBody": {
#                 "required": True,
#                 "content": {
#                     media: {"schema": {"type": "string", "format": "binary"}}
#                     for media in ("image/jpeg", "image/png", "application/octet-stream")
#                 },
#             }
#         },
#     )
#     async def predictions(request: Request) -> PredictionResponse:
#         """Send image bytes as the body. Boxes refer to the EXIF-oriented image."""
#         media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
#         if media_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
#             raise HTTPException(415, "Send JPEG or PNG bytes as the request body")
#         content = bytearray()
#         async for chunk in request.stream():
#             if len(content) + len(chunk) > config.max_upload_bytes:
#                 raise HTTPException(413, "Image exceeds the upload limit")
#             content.extend(chunk)
#         image = await run_in_threadpool(decode_image, bytes(content), config.max_image_pixels)
#         try:
#             return await run_in_threadpool(detector.predict, image)
#         except ModelUnavailableError as exc:
#             raise HTTPException(503, str(exc)) from exc
#         finally:
#             image.close()
#
#     return api
#
#
# app = create_app()

from ultralytics import YOLO

print('START')

# Загрузить предварительно обученную модель YOLO (рекомендуется для обучения)
model = YOLO("configs/models/yolo26l.pt")

# Обучить модель на наборе данных 'coco8.yaml' в течение 3 эпох
results = model.train(
    data="datasets/dataset.yaml",
    epochs=100,
    batch=4, # сколько картинок за раз
    workers=0,
    cache=False, # для оптимизации
    device="mps",# для оптимизации
    save_period=20, # сохранять результат каждые 20 эпох
    # last=True, # начать с последнего чекпойнта
    amp=True, # оптимизация для чисел с плавающей точкой
    # аугментации
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.0, # отдельно посмотреть с 0.3
    mixup=0.0
)



# Оценить производительность модели на проверочной выборке
results = model.val()

# Выполнить обнаружение объектов на изображении с помощью модели
# results = model("https://ultralytics.com/images/bus.jpg")

# Экспортировать модель в формат ONNX
success = model.export(format="onnx")
