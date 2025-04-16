# Authors: Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

from pathlib import Path
import matplotlib.pyplot as plt
import time
import numpy as np

class QC:
    def __init__(self, ll_state, rejection_policy):
        """
        Initialize the QC class.

        Args:
            ll_state (ll.LosslessPipeline): The Lossless derivative state to QC
            rejection_policy (dict): The rejection policy to use for the QC
                Does not fully reject channels or ICs, but flags them as bads
        """

        self.ll_state = ll_state
        self.rejection_policy = rejection_policy
        self.post_qc_bads = []
        self.post_qc_bads_ids = []

        # Add channels to be rejected as bads
        for key in self.rejection_policy["ch_flags_to_reject"]:
            if key in self.ll_state.flags["ch"]:
                self.ll_state.raw.info["bads"] += self.ll_state.flags["ch"][key].tolist()
        self.ll_state.raw.load_data()
        self.ll_state.raw = self.ll_state.raw.set_eeg_reference('average')

        # Add ICs to be rejected
        ic_labels = self.ll_state.flags["ic"]
        mask = np.array([False] * len(ic_labels["confidence"]))
        for label in self.rejection_policy["ic_flags_to_reject"]:
            mask |= ic_labels["ic_type"] == label
        mask &= ic_labels["confidence"] > self.rejection_policy["ic_rejection_threshold"]
        flagged_ics = ic_labels.loc[mask]
        if not flagged_ics.empty:
            self.ll_state.ica2.exclude.extend(flagged_ics.index.tolist())
        
    def apply_qc(self):
        """
        Apply the QC to the Lossless state.
        """
        self.ll_state.ica2.exclude = self.post_qc_bads_ids

        cleaned_raw = self.rejection_policy.apply(self.ll_state)

        return cleaned_raw

    def run(self):
        """
        Run the QC process.

        Plots the scrolling time course of Independent Components (ICs).
        Plots topographical maps for all ICs from a Lossless state.
        Creates and monitors a .local_reject file for bad components.
        Clicking on any IC will open it in a new figure window.
        Text will be shown in red for components marked as artifacts.

        Class attributes are updated with the post-QC bads.

        To apply changes to the Lossless state, call apply_qc().
        """
        start_time = time.time()

        self._plot_all_ic_topos()
        self._plot_ic_scrollplot()

        # Time taken to QC
        end_time = time.time()
        print(f"Time taken to QC: {end_time - start_time:.2f} seconds")

        # Add some sort of post qc step that grabs .local_reject file and
        # uses it to update the ll_state.ica2.exclude list
        local_reject_file = Path('.local_reject')
        if local_reject_file.exists():
            with open(local_reject_file, 'r') as f:
                contents = f.read().splitlines()
            print('Components to reject after QC:', contents[2])
            set_str = contents[2].strip('{}').strip()
            self.post_qc_bads = [comp.strip().strip("'") for comp in set_str.split(',') if comp.strip()]
            self.post_qc_bads_ids = [int(comp.replace('ICA', '')) for comp in self.post_qc_bads]

    def _plot_ic_scrollplot(self):
        """
        Plot the scrolling time course of Independent Components (ICs).
        
        Parameters:
        -----------
        ll_state : ll.LosslessPipeline
            The loaded Lossless derivative state containing ICA information
        """
        
        # Plot scrolling time course with optimizations
        self.ll_state.ica2.plot_sources(self.ll_state.raw,
                                start=0, show=True,
                                title='IC Time Courses',
                                block=True)

    def _plot_all_ic_topos(self):
        """
        Plot topographical maps for all ICs from a Lossless state.
        Creates and monitors a .local_reject file for bad components.
        Clicking on any IC will open it in a new figure window.
        Text will be shown in red for components marked as artifacts.
        """
        from PyQt5.QtCore import QTimer, QObject, pyqtSignal
        
        class FileWatcher(QObject):
            file_changed = pyqtSignal()
            
            def __init__(self, file_path):
                super().__init__()
                self.file_path = file_path
                self.last_modified = file_path.stat().st_mtime
                
            def check_file(self):
                try:
                    current_modified = self.file_path.stat().st_mtime
                    if current_modified > self.last_modified:
                        self.last_modified = current_modified
                        self.file_changed.emit()
                except Exception as e:
                    print(f"Error monitoring file: {e}")

        # Get the number of ICs
        n_components = self.ll_state.ica2.n_components_
        
        # Create initial file with known bad components
        bad_components = self.ll_state.ica2.exclude
        
        local_reject_file = Path('.local_reject')
        with open(local_reject_file, 'w') as f:
            f.write(f"pylossless-qc-reference\n")  # First line is label
            f.write("0,0\n")  # Second line is time range
            f.write(str({f'ICA{str(item).zfill(3)}' for item in bad_components}))  # Third line is bad components
        
        # Function to read bad components and time range from file
        def get_bad_components():
            with open(local_reject_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 3:
                    return 0, 0, set()
                
                # Get time range from second line
                try:
                    xmin, xmax = map(float, lines[1].strip().split(','))
                except:
                    xmin, xmax = 0, 0
                    
                # Get bad components from third line
                if not lines[2].strip():
                    return xmin, xmax, set()
                
                # Clean up the set string and parse components
                set_str = lines[2].strip()
                # Remove the outer set brackets and any newlines
                set_str = set_str.strip('{}').strip()
                # Split by comma and clean up each component
                components = [comp.strip().strip("'") for comp in set_str.split(',') if comp.strip()]
                return xmin, xmax, set(components)

        # Create new figure with subplots
        n_cols = 5
        n_rows = (n_components + 4) // 5
        
        fig = self.ll_state.ica2.plot_components(
            picks=range(n_components),
            ch_type='eeg',
            title='IC Topographies',
            show=False,
            ncols=n_cols,
            nrows=n_rows,
        )
        
        fig.set_size_inches(15, 3 * n_rows)
        
        # Get current bad components and convert to indices
        _, _, bad_components = get_bad_components()
        bad_indices = {int(comp.replace('ICA', '')) for comp in bad_components}
        
        # Update component labels
        if hasattr(self.ll_state, 'flags') and 'ic' in self.ll_state.flags:
            ic_flags = self.ll_state.flags['ic']
            
            for idx in range(n_components):
                ax = fig.axes[idx]
                ax.set_title('')
                
                if idx in ic_flags.index:
                    ic_type = ic_flags.loc[idx, 'ic_type']
                    confidence = ic_flags.loc[idx, 'confidence']
                    # Check if this component is in the bad list
                    text_color = 'red' if idx in bad_indices else 'black'
                    ax.text(0.5, -0.1, f'IC{idx}\n{ic_type}\n{confidence:.2f}', 
                            horizontalalignment='center',
                            verticalalignment='top',
                            transform=ax.transAxes,
                            fontsize=8,
                            color=text_color)
        
        plt.subplots_adjust(bottom=0.1, hspace=0.25, wspace=0.3)
        fig.canvas.draw_idle()
        plt.show(block=False)

        def plot_scroll_difference():
            if not hasattr(plot_scroll_difference, '_called'):
                plot_scroll_difference._called = True
                plot_scroll_difference.bad_components_history = []  # Initialize history list
                return
                
            xmin, xmax, bad_components = get_bad_components()
            
            # Add current bad_components to history
            plot_scroll_difference.bad_components_history.append((time.time(), bad_components))
            
            # Print history for debugging
            print("\nBad components history:")
            for timestamp, components in plot_scroll_difference.bad_components_history:
                print(f"Time: {timestamp:.2f}, Components: {components}")
            
            # Create a fresh copy of the full raw data each time and load it into memory
            snap_state = self.ll_state.raw.copy().load_data()
            # Apply ICA exclusions
            self.ll_state.ica2.exclude = [int(comp.replace('ICA', '')) for comp in bad_components]
            self.ll_state.ica2.apply(snap_state)
            snap_state.set_eeg_reference('average')
            
            # Window the data after cleaning
            snap_state = snap_state.crop(tmin=xmin, tmax=xmax)
            
            # Create a single figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Calculate sample points more precisely
            sfreq = self.ll_state.raw.info['sfreq']
            start_idx = int(xmin * sfreq)
            end_idx = int(xmax * sfreq)
            
            # Extract data for the time window, ensuring we get the full range
            data = snap_state.get_data()  # Get all data from the cropped state
            raw_data = self.ll_state.raw.get_data(start=start_idx, stop=end_idx)
            
            # Create time arrays
            n_samples_clean = data.shape[1]
            n_samples_raw = raw_data.shape[1]
            times = np.linspace(xmin, xmax, n_samples_clean)
            raw_times = np.linspace(xmin, xmax, n_samples_raw)
            
            ch_names = self.ll_state.raw.ch_names
            n_channels = len(ch_names)
            
            # Create y-axis positions for the channels
            positions = np.arange(n_channels) * 3
            positions = positions[::-1]  # Reverse order to match traditional EEG display
            
            # Plot raw data first (transparent)
            for i, ch_name in enumerate(ch_names):
                channel_data = raw_data[i]
                normalized_data = channel_data / (np.max(np.abs(channel_data)) + 1e-6)
                ax.plot(raw_times, normalized_data + positions[i], 
                    label=ch_name + ' (raw)', 
                    linewidth=0.8,
                    color='#E69F00')  # Seaborn-style orange
            
            # Plot cleaned data on top (opaque)
            for i, ch_name in enumerate(snap_state.ch_names):
                channel_data = data[i]
                normalized_data = channel_data / (np.max(np.abs(channel_data)) + 1e-6)
                ax.plot(times, normalized_data + positions[i],
                    label=ch_name + ' (cleaned)',
                    linewidth=0.8,
                    color='#009E73')  # Seaborn-style green
            
            # Set y-axis ticks and labels
            ax.set_yticks(positions)
            ax.set_yticklabels(ch_names)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channels')
            ax.set_title('EEG Data Comparison (Raw vs Cleaned)')
            ax.grid(True)
            ax.legend().set_visible(False)

            # Add key event handler for Escape key
            def on_key(event):
                if event.key == 'escape':
                    plt.close(fig)
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            plt.tight_layout()
            plt.show()
            
            return

        # Set up Qt-based file monitoring
        watcher = FileWatcher(local_reject_file)
        watcher.file_changed.connect(plot_scroll_difference)
        
        # Create timer in the main Qt thread
        timer = QTimer()
        timer.timeout.connect(watcher.check_file)
        timer.start(500)  # Check every 500ms
        
        # Store timer and watcher as figure properties to prevent garbage collection
        fig.timer = timer
        fig.watcher = watcher
        
        # Define click event handler
        def on_click(event):
            if event.inaxes:
                ax_idx = fig.axes.index(event.inaxes)
                if ax_idx < n_components:
                    # Create PSD plot for the component
                    self.ll_state.ica2.plot_properties(
                        self.ll_state.raw,
                        picks=[ax_idx],
                        dB=True,
                        plot_std=True
                    )
                    plt.show()

        fig.canvas.mpl_connect('button_press_event', on_click)
        return fig
