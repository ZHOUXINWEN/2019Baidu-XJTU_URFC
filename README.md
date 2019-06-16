# 2019Baidu-XJTU_URFC
2019Baidu&amp;XJTU_URFC Preliminary Round Code


GeResult_ensemble.py  对训练得到的模型进行融合
MM_val.py             在验证集上分析训练得到的单模型
MM_val_ensemble.py    在验证集上分析训练得到的多模型， 里面的confusion matrix可以分析分类结果

multimodal_train_val.py   训练模型

data/Split_TrainVal.py      拆分训练验证集
data/select.py              去除黑块面积超过25%的图片
data/train/resample_txt.py  重新进行采样使得每个类别数量一致


2019Baidu-XJTU_URFC/MultiModal_BDXJTU2019.py   里面还包含Test Time Augmentation(TTA) 的输出


visit2array_adapt.py  将visit txt数据转化成npy (Reference: https://github.com/czczup/UrbanRegionFunctionClassification)
