# 考核项目

伪装目标分析是一个具有重要理论与应用价值的问题。本项目以机器学习核心方法为工具，实现伪装目标的检测和分割

# 考核任务

考核须完成以下三个任务：
任务 1: 伪装目标分割。给定图片，利用条件随机场(CRF)模型分割图片中的伪装目标，即输出图片中伪装目标的像素, 计算结构度量 Structure-measure 和平均绝对误差 Mean Absolute Error 指标。特征可自己设计。

任务 2: 伪装目标分割。给定图片，任选一种模型（不同于任务 1 中的模型）分割图片中的伪装目标，即输出图片中伪装目标的像素, 计算结构度量 Structure-measure 和平均绝对误差 Mean Absolute Error 指标。并与问题 1 中的结果进行比较分析。

任务 3: 伪装目标检测集成学习。给定图片，利用 Adaboost 方法检测图片中的伪装目标，即输出每一张图片中伪装目标的 Bounding Box,画出 PR 曲线并计算 AP(Average Precision)指标。基学习器和特征可自己设计。
