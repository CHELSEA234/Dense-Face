## Dense annotation scheme
We use the mediapipe to generate dense annotation for the T2I-Dense-Face dataset. The quick view on the code structure:
```bash
./Dense-Face
    └──  annotation_toolbox
           ├── dense_annotation_demo.ipynb (Crop the SD output image and produce the face region mask)
           ├── Dense-Face-mediapipe.yml (env. file for the annotation)
           ├── alignedFaceParsingUtils.py 
           ├── requirement.txt (in case the yml file fails)
           └── ... (various mediapipe models)
```

### Get started 
- To create your environment by
  ```
  conda env create -f Dense-Face-mediapipe.yml
  ```
  it uses `python 3.7` yet the main Dense-Face uses `python 3.7`. If unfortunately yml does not work, please 
  ```
  pip install -r requirement.txt
  ```
- Check out `./dense_annotation_demo.ipynb`. 

<!-- ### Reference
If you think our work is helpful, please cite:
```Bibtex
@inproceedings{hifi_net_xiaoguo,
  author = { Xiao Guo and Xiaohong Liu and Zhiyuan Ren and Steven Grosz and Iacopo Masi and Xiaoming Liu },
  title = { Hierarchical Fine-Grained Image Forgery Detection and Localization },
  booktitle = { CVPR },
  year = { 2023 },
}
``` -->