## Introduction

The Lossless pipeline is an EEG data preprocessing technique that relies on independent component analysis (ICA). The ICA is used to isolate arefacts into independent components that can be marked for rejection. Additionally, cortical components can be further investigated in more complex types of analyses. Despite the automated nature of the technique up to this point, the decisions made by the pipeline need to be manually reviewed. Quality control (QC) is the procedure of manually reviewing data.

During QC, a reviewer will be interacting with the annotations created by the pipeline. These annotations are split up into three groups: channel, component, and time annotations. Reviewing them is accomplished by scrolling through the data and either making changes to the pipeline's decisions, or adding additional annotations. These decisions are made by the main [pipeline scripts](https://github.com/BUCANL/EEG-IP-L/wiki/Pipeline-Scripts) such as **s01** and are customized by their accompanying batch configuration files.

## Artefacts

In EEG analysis, an artefact is any period of activation that does not represent cortical signal. Artefacts can be biological (eye blinks, lateral eye movements, muscle movements, hearbeat, etc.) or environmental (screen noise, channel noise from bad electrodes, etc.). Most artefacts can be identified based on unusually low/high frequencies or amplitudes compared to those of typical cortical signal.

![QC Artefact Example](fig/qc_marks_example.png "QC Artefact Example")

This figure is an example of the EEG **component** data scroll plot. During the time period of 542-543 seconds, there are several components with an artefact. Note that these artefacts have unusually high amplitudes compared to cortical data and are seen as large spikes. Artefacts can be removed from the data by marking the components or time periods for removal. The following lessons will describe how to interact with the data and how to make decisions about removing artefacts.

## Pipeline Annotations

The annotations created as part of the Lossless pipeline are designed with purging of artefactual channels, components, and time in mind. This is accomplished using the manual mark. The manual mark is an annotation that is consistent across all three annotation structures (channel, component, and time). The purpose of this mark is to allow for selecting data for removal. The manual mark is structured such that it is the combination of all other decision criteria. For example, if a channel is found to be bridged to its neighbour, it would have both the "bridge" and "manual" mark. Channels marked as manual are the set of channels that contain bad or potentially problematic data. Thus, purging manually marked data across channels, components, and time will remove data that has been marked as artefactual for any reason.

The only manual marks that can be adjusted during QC are component and time. Channel manual marking is determined by parameters in the batch configuration file associated with **s01**. Changing decisions made by the pipeline is covered in a further lesson.

> ## Purging Manual Marks
> During purging, manually marked channels or components are removed for the entire datafile. However, removing manually marked time inserts boundary events that cause rejection of trials during segmentation.
>
> {: .source}
{: .callout}

## Channel Pipeline Annotations 

Channel labels produced by the Lossless pipeline are created via the **s01** script and can be customized by its associated batch configuration file. Below is a table with a brief description of each mark type:

| Label | Description |
| ------ | --------------- |
| manual | The label produced by pipeline decisions along with other labels. It is typically used to indicate which channels are considered artefactual for any reason. |
| ch_sd | Pipeline decision flag indicating that channels were too often outliers compared to other channels for the measure of standard deviation of voltage within one second epochs. |
| low_r | Pipeline decision flag indicating that channels were too often outliers compared to other channels for the measure of correlation coefficient to spatially neighbouring channels within one second epochs. |
| bridge | Pipeline decision flag indicating that channels were outliers in terms of having high and invariant correlation coefficients to spatially neighbouring channels. |
| rank | Pipeline decision identifying the channel that has the least amount of unique information (highest average correlation coefficient to spatially neighbouring channels) to be ignored by the independent component analysis to account for the rank deficiency of the average referenced data. |

## Component Pipeline Labels

As part of **s05**, each component is evaluated via the [ICLabel](https://labeling.ucsd.edu/tutorial/format) plugin. Components that are identified to be primarily artefactual (i.e. eye. channel noise, muscle, and heart) are marked as manual.

| Label | Description |
| ------ | -----------------|
| manual | The interactive label modified by analysts interactively or by a pipeline decision along with other labels. It is typically used to indicate which components are considered artefactual for any reason. |
| ic_rt | Pipeline decision flag indicating that components were not replicated across the three parallel AMICA runs. |
| brain | ICLabel mark indicating that the component has cortical characteristics. |
| muscle | ICLabel mark indicating that the component has EMG characteristics. |
| eye | ICLabel mark indicating that the component has EOG characteristics. |
| heart | ICLabel mark indicating that the component has ECG characteristics. |
| line_noise | ICLabel mark indicating that the component has electrical mains noise characteristics. |
| chan_noise | ICLabel mark indicating that the component has channel independence characteristics. |
| other | ICLabel mark indicating that the component could not be confidently classified within any other ICLabel classification. |
| ambiguous | QC procedure rater markup indicating that the component is difficult to classify as either artefact or not. |

## Time Pipeline Labels

The EEG component and channel data windows display marks throughout the file that relate to time. The time marks are **independent of the y-axis**, i.e. these marks are only related to time and are not related to the location of the components. When the pipeline has marked time as artefactual, there will be a coloured mark that indicates why that time period is considered messy, as well as a gray manual mark.

Throughout the whole file there will be an orange bar which is the log likelihood of the AMICA decomposition. The colour of this bar at each time point relates to how much that time point influenced the overall ICA model. A transparent region means a lower contribution from this time point to the overall amount of learning done. The orange AMICA bar will not be present at some time points that are marked as artefactual because these time points did not run through AMICA.

| Label | Description |
| -------- | ----------------|
| manual | The interactive label modified by analysts interactively or by a pipeline decision along with other labels. It is typically used to indicate which time points are considered artefactual for any reason. |
| init_ind | Continuous variable indicating the initial time point index within the session. |
| mark_gap | Pipeline decision flag indicating that time points are within a short gap between other annotations. | 
| ch_sd | Pipeline decision flag indicating that time points were too often outliers across channels compared to other time points for the measure of standard deviation of voltage within one second epochs. |
| low_r | Pipeline decision flag indicating that time points were too often outliers across channels compared to other time points for the measure of correlation coefficient to spatially neighbouring channels within one second epcohs. |
| logl_init | Log likelihood of initial AMICA decomposition. |
| ic_sd1 | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component standard deviation of voltage within one second epochs. |
| logl_A | Log likelihood of final AMICA decomposition (replication A). |
| logl_B | Log likelihood of final AMICA decomposition (replication B). |
| logl_C | Log likelihood of final AMICA decomposition (replication C). |
| ic_sd2 | Pipeline decision flag indicating that time points were too often outliers across final components compared to other time points for the measure of component standard deviation of voltage within one second epochs. |
| ic_dt | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component spectral Theta within one second epochs. |
| ic_a | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component spectral Alpha within one second epochs. |
| ic_b | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component spectral Beta within one second epochs. |
| ic_lg | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component spectral Low Gamma within one second epochs. |
| ic_hg | Pipeline decision flag indicating that time points were too often outliers across initial components compared to other time points for the measure of component spectral High Gamma within one second epochs. |

{% include links.md %}

