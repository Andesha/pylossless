---
title: "Introdution to Lossless Parameters"
teaching: 35
exercises: 10
questions:
- "What are parameters?"
- "How does the Lossless pipeline use parameters to make decisions?"
objectives:
- "Understand what parameters are and how they are used in the Lossless pipeline."
keypoints:
- "Parameters in batch configuration files can be edited to change the decisions made in the pipeline scripts."
- "Editing parameters is the only way to change the decisions the pipeline has made about channels."
---

## Introduction

Each script in the Lossless pipeline has parameters. Parameters can be conceptualized as various sliders and dials for tuning the performance of a particular script. In the case of Lossless, this is often montage information, path specifications, memory and time allocations on the remote, and outlier detection decision criteria. Parameters in Lossless are controlled (and defined) by the associated batch configuration files. 

It is highly recommended that parameters related to pipeline decision criteria are optimized for each study. In the case of montage information and staging, it is *required* to change the configuration files. The parameters for channel and time decision criteria as well as montage information and staging can be edited in batch configuration file c01. Batch configuration files c03 and c05 also contain parameters, however these parameters should only be edited by expert users and are not covered in this tutorial.

## Channel Decision Criteria

This figure shows the decision criteria for removing artefactual **channels** during the pipeline. These decisions are based on statistical distributions of the data and the pipeline decision criteria parameters. The parameters can be edited to change the decisions the pipeline is making about channels. Decisions regarding removing channels are made by the pipeline based on the parameters and **can not** be edited during the quality control procedured by the reviewer.

![Decision Criteria Channels]({{ page.root }}/fig/PipelineChannelCriteria.png "Decision Criteria Channels")

**A.** The voltage variance for each channel calculated for every 1-second epoch.

**B.** The median, quantiles, and outliers for each 1-second epoch across all of the channels for the measure of voltage variance. In this example, a channel is marked as an outlier if the channel's voltage variance is greater than 6 times the median to quantile distance. The multiplication factor for the median to quantile distance (6 in this example) is a parameter in `c01` that can be edited.

**C.** Binary (yes/no) decision for each channel at every 1-second epoch based on if the channel is an outlier during that epoch. Channels that are an outlier are shown in yellow.

**D.** Plot that shows the percentage of time that each channel is an outlier. The red line is the critical cut-off value and if a channel is an outlier for a greater percentage of time than the critical cut-off it will be marked for removal. The crtical cut-off value is a parameter in `c01` that can be edited. In this example the critical cut-off value is [0.2] indicating that a channel must be an outlier for 20% of the epochs to be marked for removal.

## Channel Decision Parameters

In the batch configuration file `c01`, the parameters that begin with `sd_ch_` are related to the channel decision criteria for voltage variance.

The **[sd_ch_o]** parameter is the factor to multiple against the median to quantile distance to determine outliers. Increasing this value will increase the critical distance for outlier detection, meaning that for every channel, less epochs will be considered outliers.

The **[sd_ch_f_o]** parameter is the critical cut-off and indicates what percentage of epochs have to be an outlier for each channel for that channel to be considered artefactual. Increasing this value will mean that a greater percentage of epochs have to be an outlier for that channel to be marked for removal. Increasing this value results in less channels being considered artefactual.

If there are non-artefactual channels that are being marked for removal for the measure of voltage variance, it is recommended to try increasing the **[sd_ch_o]** and/or the **[sd_ch_f_o]** parameters.

The **[sd_ch_vals]** parameter are the quantiles used for the critical distance. The value for this parameter in the above example is [.3 .7], indicating that the 30% and 70% quantiles are used. When optimizing parameters it is recommended to leave this parameter at the default values and change the **[sd_ch_o]** and **[sd_ch_f_o]** parameters to change pipeline decisions.  


## Time Decison Criteria

This figure shows the decision criteria for removing artefactual **time** during the pipeline. These decisions are based on statistical distributions of the data and the pipeline decision criteria parameters. The parameters can be edited to change the decisions the pipeline is making about time periods. The time decision criteria is the same idea as the channel decision criteria but the data is collasped in the other direction. 

![Decision Criteria Time]({{ page.root }}/fig/PipelineTimeCriteria.png "Decision Criteria Time")

**A.** The voltage variance for each channel calculated for every 1-second epoch.

**B.** The median, quantiles, and outliers for each 1-second epoch across all of the time points for the measure of voltage variance. In this example, a time point is marked as an outlier if the time point's voltage variance is greater than 6 times the median to quantile distance. The multiplication factor for the median to quantile distance (6 in this example) is a parameter in `c01` that can be edited.

**C.** Binary (yes/no) decision for each time point at every channel based on if the time point is an outlier for that channel. Time points that are an outlier are shown in yellow.

**D.** Plot that shows the percentage of channels that are outliers for each time point. The red line is the critical cut-off value and if a time point is an outlier for a greater percentage of channels than the critical cut-off it will be marked for removal. The crtical cut-off value is a parameter in `c01` that can be edited. In this example the critical cut-off if [0.2] indicating that at any given time point, 20% of channels must be an outlier for the time point to be identified as artefactual.
 
## Time Decision Parameters

In the batch configuration file `c01` the parameters that begin with `sd_t_` are related to the time decision criteria for voltage variance.

The **[sd_t_o]** parameter is the factor to multiple against the median to quantile distance to determine outliers. Increasing this value will increase the critical distance for outlier detection. This means that at every time point, less channels will be considered an outlier.

The **[sd_t_f_o]** parameter is the critical cut-off and indicates what percentage of channels have to be an outlier at each time point for that time point to be considered artefactual. Increasing this value will mean that a greater percentage of channels have to be an outlier for that time point to be marked for removal. Increasing this value results in less time points being considered artefactual. 

If there are non-artefactual time points that are being marked for removal for the measure of voltage variance, it is recommended to try increasing the **[sd_t_o]** and/or the **[sd_t_f_o]** parameters. 

The **[sd_t_vals]** parameter are the quantiles used for the critical distance. The value for this parameter in the above example is [.3 .7], indicating that the 30% and 70% quantiles are used. When optimizing parameters it is recommmended to leave this parameter at the default values and change the **[sd_t_o]** and **[sd_t_f_o]** parameters to change pipeline decisions.  


{% include links.md %}

---
