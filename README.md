
# Encodeep

The whole pipeline consists of a total of three distinct steps. The first step involves transforming the original multivariate SITS data into 2D images leveraging four distinct TS encoding methods. The second step is devoted to concatenating the different encoded information with the aim to get a unique representation. It refers to combining the images of the four encoding techniques into a single image. The third and last phase are devoted to performing the final decision, which is implemented via a CNN model.

![Screenshot-113](https://github.com/aazzaabidi/Encodeep/assets/73762433/d01df86f-3849-4f4a-88d7-ce8e656d15db)

## Data Preparation 

### Requirements

- Python 3.x
- NumPy
- scikit-learn
- PyTorch
- PyTorch Image
- PyWavelets
- TensorFlow 2.x
- Pandas
- Seaborn

### Data Preparation

Before running the main script, ensure that the satellite image time series data is prepared for transformation and classification using **data_preparation.py**. This file contains Python code to transform satellite image time series data using various transformation techniques such as Gramian Angular Field (GADF), Gramian Angular Summation Field (GASF), Markov Transition Field (MTF), and Recurrence Plot (RP). The transformed data is split into train, test, and validation sets using 5-fold cross-validation and saved as numpy arrays. This includes:

1. **Data Splitting**: Split the data into train, test, and validation sets. Each set should be stored in separate folders.

2. **Data Encoding**: Encode the data into 2D images suitable for transformation. Each time series should be transformed into a 2D image representation.

3. **Label Encoding**: Encode the labels corresponding to the data samples. Ensure that each label corresponds to the correct data sample.

### Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/aazzaabidi/Encodeep.git
    ```

2. Ensure that the data preparation steps mentioned above have been completed.

4. Update the `data_folder` and `labels_folder` variables in the script with the paths to your prepared data and labels folders, respectively.

5. Run the script:

    ```bash
    python data_preparation.py
    ```

6. The transformed data will be saved in the specified directory.

### Files

- `transform_data.py`: Python script to transform the satellite image time series data and split it into train, test, and validation sets.
- `Techniques.py`: Python module containing classes for implementing transformation techniques such as GADF, GASF, MTF, and RP.

## Models

1. **InceptionTime**: An implementation of the InceptionTime model.
2. **MultiRocket**: An implementation of the MultiRocket model.
3. **ResNet50**: An implementation of the ResNet50 model.
4. **RandomForest**: An implementation of the RandomForest classifier.
5. **TempCNN**: An implementation of the TempCNN model.

## References

- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)

--- 

To cite this work please refer to :
@article{abidi2023,
  title={Combining 2D encoding and convolutional neural network to enhance land cover mapping from Satellite Image Time Series},
  author={Abidi, Azza and Ienco, Dino and Abbes, Ali Ben and Farah, Imed Riadh},
  journal={Engineering Applications of Artificial Intelligence},
  volume={122},
  pages={106152},
  year={2023},
  publisher={Elsevier}
}
