import numpy as np

from meshbestfit.labels import class_index, classes_txt, yolo_text
from meshbestfit.schema import Category

CATS = [Category(id=1, name="a"), Category(id=3, name="b")]


def test_class_ids_contiguous_or_offset():
    assert class_index(CATS, None) == {1: 0, 3: 1}
    assert classes_txt(CATS, None) == "a\nb\n"
    assert class_index(CATS, -1) == {1: 0, 3: 2}  # repo parity: COCO id - 1
    assert classes_txt(CATS, -1) == "a\n\nb\n"  # line number == class id


def test_yolo_and_poly_text():
    rects = np.array([[100.0, 50.0, 300.0, 150.0]])
    assert yolo_text(rects, (400, 200), [2], poly=False) == "2 0.500000 0.500000 0.500000 0.500000\n"
    assert yolo_text(rects, (400, 200), [2], poly=True) == (
        "2 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000\n"
    )
