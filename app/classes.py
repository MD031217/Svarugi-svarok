from app.schemas import DatasetClass

# IDs follow train/classes.txt; descriptions follow source presentation slide 40.
DATASET_CLASSES = (
    DatasetClass(id=0, name="adj", description="Прилегающие дефекты"),
    DatasetClass(id=1, name="int", description="Дефекты целостности"),
    DatasetClass(id=2, name="geo", description="Дефекты геометрии"),
    DatasetClass(id=3, name="pro", description="Дефекты постобработки"),
    DatasetClass(id=4, name="non", description="Дефекты невыполнения"),
)
