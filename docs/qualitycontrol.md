# Introduction to the Lossless Quality Control

Temp

## TODO Notes

* how to launch
* how to interact
* various figures and screenshots of what they look like
* adding time back in
* taking time out
* adding components in/out
* exiting and applying
* move figures around

<span style="color:red">A.</span> A window that displays the **component** EEG data. This is the window that you will be interacting with as you QC. You will be making your decisions in this window by adding or removing a manual mark for components or time points. Components are sorted by the percent data variance accounted for, with the top components accounting for a greater percentage of the channel data. To scroll through the data, use the `<<` and `>>` buttons in this window. These buttons will scroll both the component EEG data and the channel EEG data windows (Figure C). Your decisions can be saved by clicking the `| Update EEG Structure |` button in the **component** EEG data window.

<span style="color:green">B.</span> A figure that displays the ICLabel classification breakdown for each component. This figure will not be loaded when running the `qc_lite.htb` script.

<span style="color:blue">C.</span> A window that shows the **channel** EEG data. It also contains an overlay feature that can be toggled on/off or updated while quality controlling. Use of this feature will be explained in subsequent lessons.

<span style="color:yellow">D.</span> A figure that displays an array of squares corresponding to each 1-second epoch for each component. Each square is colored based on its activation difference from the mean. This figure will not be loaded when running the `qc_lite.htb` script.

<span style="color:violet">E.</span> Window(s) with a topography for each component. The number label for each topography can be clicked to gain more information.

<span style="color:orange">F.</span> Upon clicking a number label for a component a figure appears which displays the component's spectrum, dipole location, and a mini scroll plot of the full waveform of the selected component. More information on how to interpret this figure can be found in the [ICLabel Tutorial](https://labeling.ucsd.edu/tutorial/format).
