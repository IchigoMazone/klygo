# Integration workflows

These guides connect Klygo modules on labeled real images. Generated datasets, weights, and reports stay outside code directories under `artifacts/integration/`.

| Workflow | Example | Notebook |
| --- | --- | --- |
| [Local Detector Evaluation](local-model-workflow.md) | [Run](../../examples/integration/local_model_workflow.py) | [Docs](local-model-workflow.md) |
| [Torchvision on COCO15](torchvision-coco15.md) | [Run](../../examples/integration/torchvision_coco15.py) | [Open](../../tutorials/integration/Torchvision_COCO15.ipynb) |

## COCO15 Class Notebooks

Each notebook evaluates one object class with the pinned 128-image COCO15 dataset (`artifacts/integration/coco15`) and Klygo pipeline on Colab GPU or CPU runtime:

| ID | Class | Positive images | Notebook |
| ---: | --- | ---: | --- |
| 0 | person | 61 | [Open](../../tutorials/integration/Klygo_COCO15_00_person.ipynb) |
| 1 | car | 12 | [Open](../../tutorials/integration/Klygo_COCO15_01_car.ipynb) |
| 2 | traffic light | 4 | [Open](../../tutorials/integration/Klygo_COCO15_02_traffic_light.ipynb) |
| 3 | elephant | 4 | [Open](../../tutorials/integration/Klygo_COCO15_03_elephant.ipynb) |
| 4 | umbrella | 4 | [Open](../../tutorials/integration/Klygo_COCO15_04_umbrella.ipynb) |
| 5 | handbag | 9 | [Open](../../tutorials/integration/Klygo_COCO15_05_handbag.ipynb) |
| 6 | bottle | 6 | [Open](../../tutorials/integration/Klygo_COCO15_06_bottle.ipynb) |
| 7 | wine glass | 5 | [Open](../../tutorials/integration/Klygo_COCO15_07_wine_glass.ipynb) |
| 8 | cup | 10 | [Open](../../tutorials/integration/Klygo_COCO15_08_cup.ipynb) |
| 9 | knife | 7 | [Open](../../tutorials/integration/Klygo_COCO15_09_knife.ipynb) |
| 10 | spoon | 5 | [Open](../../tutorials/integration/Klygo_COCO15_10_spoon.ipynb) |
| 11 | bowl | 9 | [Open](../../tutorials/integration/Klygo_COCO15_11_bowl.ipynb) |
| 12 | chair | 9 | [Open](../../tutorials/integration/Klygo_COCO15_12_chair.ipynb) |
| 13 | book | 6 | [Open](../../tutorials/integration/Klygo_COCO15_13_book.ipynb) |
| 14 | teddy bear | 6 | [Open](../../tutorials/integration/Klygo_COCO15_14_teddy_bear.ipynb) |

