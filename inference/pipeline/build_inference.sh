kernel_name="rapids"
echo "--- Install jupyter notebook kernel: ${kernel_name} ---"
conda create -n ${kernel_name} python=3.8.15 -y
source activate ${kernel_name}
conda install -y ipykernel
ipython kernel install --user --name=${kernel_name}
python3 -m pip install -U pyarrow==11.0.0 fastparquet==2023.2.0 gcsfs==2023.3.0 xgboost==1.7.0 dask-ml==2023.3.24 scikit-learn==1.2.2 papermill pandas==1.5.3 matplotlib nbconvert nbformat bokeh seaborn

echo "Running: inference.py $@"
python inference.py $@