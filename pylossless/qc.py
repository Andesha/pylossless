# Authors: Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

import mne

class QC:
    def __init__(self, ll_state, rejection_policy):
        self.pipeline = ll_state
        self.raw = ll_state.raw
        self.rejection_policy = rejection_policy

        raise  ValueError('Not implemented')
    def run(self):
        raise  ValueError('Not implemented')

    def _plot_ic_scrollplot(self, ll_state, picks=None):
        """
        Plot the scrolling time course of Independent Components (ICs).
        
        Parameters:
        -----------
        ll_state : ll.LosslessPipeline
            The loaded Lossless derivative state containing ICA information
        picks : list or None, optional
            List of component indices to plot. If None, plots all components.
        """
        # If no specific components are selected, plot all
        if picks is None:
            picks = range(ll_state.ica2.n_components_)
        
        # Plot scrolling time course with optimizations
        ll_state.ica2.plot_sources(ll_state.raw, picks=picks,
                                start=0, show=True,
                                title='IC Time Courses',
                                block=True)
