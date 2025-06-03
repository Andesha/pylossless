The EEG component data window is the main window you will be interacting with during the procedure. Below is a table containing all of the actions that are possible during QC.

| Action | Instructions |
| --------- | ------------|
| **Removing or adding back in a component** | Components that are marked for removal will be "grayed out" and will have a gray flag beside the component on the left hand side of the component data window. To toggle the state of the manual mark on a component, right click it. |
| **Marking out time** | To reject a given period of time, begin by left clicking the beginning of the artefactual time period. Move your mouse to the end of the artefactual time period and left click again. This will add a green highlight to the selected time period. Next, hover the mouse over the highlighted time period and click **'m'** on the keyboard to add the gray manual mark. |
|  **Adding time back in** | Time that the pipeline marked for removal can also be added back in. If you hover over the selected area that you want to add back in and press **[shift + m]**, the manual marking for this time period will be discarded. |
| **Marking a component ambiguous** | The ambiguous mark is used to identify components that are difficult to classify as artefactual. To mark a component as ambiguous, hover over it, and press **a**. The component shopuld now appear to be blue. To remove this mark, repeat the process. |
| **Using the overlay** | To better visualize changes to the scalp data, the EEG **channel** figure can display an overlay. To enable this feature, click into the EEG **channel** window to make it your current focus, and press **o**. Data in blue, is the EEG data with no artefacts removed. Data in gray, reflects the changes made by rejecting various components. To update this figure after toggling components, make the EEG **channel** window your focus by clicking into it, and pressing **u**. This feature is often used to confirm if a particular artefact is contained in a given component. The overlay can be turned off by pressing the **'o'** button, with the EEG **channel** window as your focus. |

> ## Note
> The number of channels or components displayed can be changed by navigating to the drop down menu under **Settings->Number of Channels to display**.
>
> {: .source}
{: .callout}

{% include links.md %}

