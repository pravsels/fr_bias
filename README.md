# Face Recognition Bias Mitigation 


## Conda setup 

Create conda environment 
```
conda create -n fr_bias python==3.10
```

Install requirements
```
pip install -r requirements.txt 
```


## Running skin analysis scripts 

To get face mask segmentations for a dataset: 
```
cd skin_analysis

python run_deeplab.py 
```

The dataset path is set as the `dataset_root` variable in `main()`. 


Following this, we can now create color distance plots, color histogram plots and color tables using the scripts in `skin_analysis` folder. 

