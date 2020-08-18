## A Strong Baseline of Pedestrian Attribute Recognition



Considering the big performance gap of various SOTA baseline, we provide a solid and strong baseline for fair comparison.



## Dependencies

- pytorch 1.4.0
- torchvision 0.5.0
- tqdm 4.43.0
- easydict 1.9


## Tricks
- sample-wise loss not label-wise loss
- big learning rate combined with clip_grad_norm
- augmentation Pad combined with RandomCrop
- add BN after classifier layer


## Performance Comparision

### Baseline Performance

- Compared with baseline performance of MsVAA, VAC, ALM, our baseline make a huge performance improvement.
- Compared with our reimplementation of MsVAA, VAC, ALM, our baseline is better.
- We try our best to reimplement [MsVAA](https://github.com/cvcode18/imbalanced_learning), [VAC](https://github.com/hguosc/visual_attention_consistency) and thanks to their code.
- We also try our best to reimplement ALM and try to contact the authors, but no reply received.

![BaselinePerf](https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/master/imgs/baseline.png)


![BaselinePerf](https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/master/imgs/baseline_rap2.png)


### SOTA Performance

- Compared with performance of recent state-of-the-art methods, the performance of our baseline is comparable, even better.

![SOTAPerf](https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/master/imgs/SOTA.png)


- DeepMAR (ACPR15) Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios.
- HPNet (ICCV17) Hydraplus-net: Attentive deep features for pedestrian analysis.
- JRL (ICCV17) Attribute recognition by joint recurrent learning of context and correlation.
- LGNet (BMVC18) Localization guided learning for pedestrian attribute recognition.
- PGDM (ICME18) Pose guided deep model for pedestrian attribute recognition in surveillance scenarios.
- GRL (IJCAI18) Grouping Attribute Recognition for Pedestrian with Joint Recurrent Learning.
- RA (AAAI19) Recurrent attention model for pedestrian attribute recognition.
- VSGR (AAAI19) Visual-semantic graph reasoning for pedestrian attribute recognition.
- VRKD (IJCAI19) Pedestrian Attribute Recognition by Joint Visual-semantic Reasoning and Knowledge Distillation.
- AAP (IJCAI19) Attribute aware pooling for pedestrian attribute recognition.
- MsVAA (ECCV18) Deep imbalanced attribute classification using visual attention aggregation.
- VAC (CVPR19) Visual attention consistency under image transforms for multi-label image classification.
- ALM (ICCV19) Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Speciﬁc Localization.


## Dataset Info
VSP 自建

## Get Started

3. Prepare datasets to have following structure:
    ```
    ${project_dir}/data
        vsp
            images/
            vsp_data.pkl
            vsp_part.pkl
            README
    ```
4. Run the `format_xxxx.py` to generate `dataset.pkl` respectively
    ```
    python ./dataset/preprocess/format_VspPersonAtt.py
    
    ``` 
5. training
   sh trian.sh


