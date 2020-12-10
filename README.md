# CLEANIR: Controllable Attribute-Preserving Natural Identity Remover
This is the official implementation of [CLEANIR: Controllable Attribute-Preserving Natural Identity Remover](https://www.mdpi.com/2076-3417/10/3/1120)

![CLEANIR Demo](./cleanir_demo.gif)

## Requirements
- face-recognition>=1.2.3
- opencv-python>=4.1.0.25
- Keras>=2.2.4
- tensorflow-gpu>=1.13.0rc0+nv
- matplotlib>=3.0.2
- numpy>=1.14.5
- pandas>=0.23.0
- tqdm>=4.32.2
- gdown>=3.10.2
- keras-facenet>=0.1a5 (if you want to run codes for evaluation on de-identification)
- azure-cognitiveservices-vision-face>=0.4.0 (if you want to run codes for evaluation on preserving facial emotion)

### Testing environment
- Ubuntu 16.04
- Python 3.5.2
- Keras 2.2.4 (backend=TensorFlow 1.13.0-rc0)

## Usage
Please check [CLEANIR_notebook.ipynb](https://github.com/chodurkhyun/cleanir/blob/master/CLEANIR_notebook.ipynb) for testing and [CLEANIR_train_notebook.ipynb](https://github.com/chodurkhyun/cleanir/blob/master/CLEANIR_train_notebook.ipynb) for training.

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{cho2020cleanir,
  title={CLEANIR: Controllable Attribute-Preserving Natural Identity Remover},
  author={Cho, Durkhyun and Lee, Jin Han and Suh, Il Hong},
  journal={Applied Sciences},
  volume={10},
  number={3},
  pages={1120},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
