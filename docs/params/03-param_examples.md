---
title: "Making Decisions About Parameters"
teaching: 15
exercises: 15
questions:
- "How do you identify incorrect parameters?"
objectives:
- "Understand how to identify and optimize incorrect parameters."
keypoints:
- FIXME
---

## Determining Optiminal Parameters

The following examples are datasets that have been run through the pipline with incorrect parameters. In each example, determine what parameters are incorrect and how they should be corrected. Remember to look at decisions the pipeline has made about removing channels and time, as well ensure the correct montage was used. 

## Examples

> ## Example 1
> 
>  ![Parameters Example 1 Channels]({{ page.root }}/fig/ParamsEx1Left.png "Parameters Example 1 Channel")
>  ![Parameters Example 1 Components]({{ page.root }}/fig/ParamsEx1Middle.png "Parameters Example 1 Components")
>  ![Parameters Example 1 Topos]({{ page.root }}/fig/ParamsEx1Right.png "Parameters Example 1 Topos")
>
> {: .source}
>
> > ## Identifying Incorrect Parameters
> > 
> > - **For Channels:** In this example, too many channels are being removed for low_r and ch_sd. Both of these marks are being added to non-artefactual channels, indicating that the parameters are not optimized. When scrolling through the data, channels that are being removed for low_r or ch_sd should be visably artefactual. 
> > For example, in this figure there are a few channels at the bottom that are truly artefactual. These are channels that we would want to be marked for removal. In this example there are also a lot of channels at the top of the file that are being marked for removal because they were outliers on the basis of channel standard deviation of voltage measure (ch_sd). These channels contain eyeblinks but are not otherwise artefactual. These are channels that we would want to remain in the data for the ICA because we know the eyeblink artefacts can be isolated into one component and be removed. 
> > 
> > - **For Time:** In this one screenshot there were no decisions that the pipeline made about time. We can agree that there weren't any artefactual time periods that should have been marked for removal.
> >
> > - **For Topographies:** The topographies in this example are warped and labeled correctly. The easiest way to determine that the topographies are correct is by looking in the component data scroll plot for the component that contains eyeblinks. The topography for this eye component should have activation at the front of the head and be labeled as an eye component. In this example we can identify component 1 as containing eyeblinks and we can see in the topography that the activation is at the front of the head and that it has been correctly labeled as eye.  
> >
> > {: .output}
> {: .solution}
>
> > ## Determining Optiminal Parameters
> >
> > - **For Channels:** The parameters for low_r and ch_sd for channels should be edited so the pipeline makes more accurate decisions about which channels are artefactual. The low_r marker identifies channels that have a low correlation coefficient to neighbouring channels throughout the file. The main parameters that can be edited to change low_r decisions are the **[r_ch_o]** and **[r_ch_f_o]** parameters. 
> > 
> > The **[r_ch_o]** parameter is the threshold for identifying outlier channels during the fixed neighbour correlation criteria. This parameter is the multiplication factor to determine how many quantiles beyond the median a channel must be for it to be indetified as an outlier. A higher value indicates that for a channel to be marked an outlier, the channel must be further from the median. 
> > 
> > The **[r_ch_f_o]** parameter is the threshold for flagging channels during the fixed neighbour correlation criteria and is used to determine what percentage of the file a channel has to be an outlier for it to be marked. A higher value indicates that the channel has to be an outlier for a greater percentage of the file to be marked.
> > 
> > Increasing the value of either the [r_ch_o] or [r_ch_f_o] parameters would decrease the number of channels that are being marked for low_r. Since in this example, non-artefactual channels are being marked we want less to be marked. 
> > B
> >
> > {: .output}
> {: .solution}
{: .challenge}


{% include links.md %}

---
