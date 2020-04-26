import matplotlib.pyplot as plt
import pickle
import datetime
from matplotlib import collections as mc
import numpy as np
import pandas as pd


class EventLog:
    def __init__(self, name, event_id, timestamp_id, log=None, resource_id=None):
        self.segment_count = {}

        # Names of keys in log file:
        self.event_id = event_id
        self.timestamp_id = timestamp_id
        self.resource_id = resource_id

        # Loading in a log object previously loaded:
        if not log:
            self.log = self.load(name)  # TODO implement faster loading.
        else:
            self.log = log
            self.clean_timezone()

        # Set the begin and end date.
        self.first = datetime.datetime(9999, 12, 31)
        self.last = datetime.datetime(1, 1, 1)
        # Counting the types of traces, this method also fixes first and last, initialized above.
        self.event_count = self.count_trace_types()

        # Name of the dataset.
        self.name = name

        # Initialize Performance Spectrum Data Frame:
        self.pf = pd.DataFrame()
        # Y values for plotting the segments:
        self.y_s = []
        # Initialize the maximum x at 0.
        self.x_max = 0
        # Saved filtering of segments used to plot.
        self.segments = []

    def clean_timezone(self):
        """
        Output: EventLog object with timestamps corrected for timezones for each event of each trace within the log.
        """
        for i in range(len(self.log)):
            for n in range(len(self.log[i])):
                # Add the timezone to the times so entire log is standardized:
                utc_offset = self.log[i][n][self.timestamp_id].tzinfo.utcoffset(
                    self.log[i][n][self.timestamp_id]).seconds
                self.log[i][n][self.timestamp_id] = self.log[i][n][self.timestamp_id].replace(tzinfo=None)
                self.log[i][n][self.timestamp_id] += datetime.timedelta(seconds=utc_offset)

    def save(self):
        """
        Saves pickle file of a Log object.
        """
        with open(self.name + '.dat', 'wb') as f:
            pickle.dump(self.log, f)

    @staticmethod
    def load(name):
        return pickle.load(open(name + '.dat', 'rb'))

    def count_trace_types(self):
        """
        Output: count of how often each unique trace occurs.
        """
        self.event_count = {}
        for i in range(len(self.log)):
            trace = (self.log[i][0][self.event_id], self.log[i][-1][self.event_id])
            if self.log[i][0][self.timestamp_id] < self.first:
                self.first = self.log[i][0][self.timestamp_id]
            if self.log[i][-1][self.timestamp_id] > self.last:
                self.last = self.log[i][-1][self.timestamp_id]

            if trace not in self.event_count:
                self.event_count[trace] = 1
            else:
                self.event_count[trace] += 1
        return self.event_count

    def count_segment_types(self):
        """
        Output: count of how often each segment in a trace occurred.
        """
        for i in range(len(self.log)):
            for n in range(0, len(self.log[i]) - 1):
                segment = (self.log[i][n][self.event_id], self.log[i][n + 1][self.event_id])
                if segment not in self.segment_count:
                    self.segment_count[segment] = 1
                else:
                    self.segment_count[segment] += 1
        self.segment_count = {k: v for k, v in reversed(sorted(self.segment_count.items(), key=lambda item: item[1]))}
        return self.segment_count

    def filter_segments(self, cutoff=0.25, compare_to='previous'):
        """
        Input: a cutoff value for deciding which segments to include as a ratio of minimum amount of occurrence
        compared to the occurrence of either the maximum or previous segment.
        Output: Frequently occurring segments according to cutoff value.
        """
        sorted_counts = self.count_segment_types()
        filtered_segments = [list(sorted_counts.keys())[0]]
        for i in range(1, len(sorted_counts.values())):
            if compare_to == 'previous':
                if list(sorted_counts.values())[i] / list(sorted_counts.values())[i - 1] >= cutoff:
                    filtered_segments.append(list(sorted_counts.keys())[i])
                else:
                    return filtered_segments
            elif compare_to == 'max':
                if list(sorted_counts.values())[i] / list(sorted_counts.values())[0] >= cutoff:
                    filtered_segments.append(list(sorted_counts.keys())[i])
                else:
                    return filtered_segments

    @staticmethod
    def batch_classifier(df, k_min=10, gamma=0, dev=1):
        """
        Input: df: Dataframe containing the start and end time of each line/(sub-)trace per segment.
        k_min: number of subsequent lines/(sub-)traces that must fulfill the batching constraints to be a batch.
        gamma: The distance allowed between subsequent lines/(sub-)traces to be considered a batch.
        dev: The amount of standard deviations the standardized times can be apart from each other;
        The lower the value, the stricter the batching; 1 <= dev <= 4
        Output: List of binary class values indicating if a line/(sub-)trace belongs to a batch.
        """
        start_batches = []
        end_batches = []
        temp_batch = []
        df_sorted = df.sort_values(by=['start_time', 'end_time'], axis=0)
        observations = df_sorted.reset_index()

        temp_batch.append(0)
        for j in range(1, len(observations)):
            if observations['end_time'][j - 1] <= observations['end_time'][j] <= gamma + observations['end_time'][
                j - 1] and observations['start_time'][j] >= observations['start_time'][j - 1]:
                temp_batch.append(j)
                if j == len(observations) - 1 and len(temp_batch) >= k_min:
                    end_batches.append(temp_batch)
            else:
                if len(temp_batch) >= k_min:
                    end_batches.append(temp_batch)

                temp_batch = []
                temp_batch.append(j)

        # batches = start_batches + end_batches
        classes = []
        for j in range(len(observations)):
            is_classified = False
            for batch in range(len(end_batches)):
                if j in end_batches[batch]:
                    classes.append(1)
                    is_classified = True
                    break

            if not is_classified:
                classes.append(0)

        temp_batch = []
        temp_batch.append(0)
        for j in range(1, len(observations)):
            if classes[j] != 1:
                if observations['start_time'][j] == observations['start_time'][j - 1] and observations['end_time'][
                    j - 1] <= observations['end_time'][j]:
                    temp_batch.append(j)
                    if j == len(observations) - 1 and len(temp_batch) >= k_min:
                        end = observations['end_time'][temp_batch[0]:temp_batch[-1] + 1].reindex(temp_batch)
                        mask = abs((end - end.median()) / end.std()) < dev
                        cleaned_temp_batch = list(end[mask].index)
                        if len(cleaned_temp_batch) >= k_min:
                            start_batches.append(cleaned_temp_batch)
            else:
                if len(temp_batch) >= k_min:
                    end = observations['end_time'][temp_batch[0]:temp_batch[-1] + 1].reindex(temp_batch)
                    start = observations['start_time'][temp_batch[0]:temp_batch[-1] + 1].reindex(temp_batch)
                    mask = abs((end - end.median()) / end.std()) < dev
                    cleaned_temp_batch = list(end[mask].index)
                    if len(cleaned_temp_batch) >= k_min:
                        start_batches.append(cleaned_temp_batch)
                temp_batch = []
                temp_batch.append(j)
        else:
            if len(temp_batch) >= k_min:
                end = observations['end_time'][temp_batch[0]:temp_batch[-1] + 1].reindex(temp_batch)
                start = observations['start_time'][temp_batch[0]:temp_batch[-1] + 1].reindex(temp_batch)
                mask = abs((end - end.median()) / end.std()) < dev
                cleaned_temp_batch = list(end[mask].index)
                if len(cleaned_temp_batch) >= k_min:
                    start_batches.append(cleaned_temp_batch)
            temp_batch = []
            temp_batch.append(j)

        for j in range(len(observations)):
            if classes[j] == 0:
                for batch in range(len(start_batches)):
                    if j in start_batches[batch]:
                        classes[j] = 2
                        break
        observations['class'] = classes
        return list(observations.sort_values(by='index')['class'])

    @staticmethod
    def classify_duration_hist(duration, num_classes):
        """
        Input: Series containing duration values for a segment, number of distinct classes to derive.
        Output: List containing the assigned class-indicators according to the histogram.
        """
        class_range = []
        for i in range(num_classes):
            if i == 0:
                start = min(duration)
            else:
                start = np.quantile(duration, (i * int(100 / num_classes) / 100))
            if i == num_classes - 1:
                end = max(duration)
            else:
                end = np.quantile(duration, ((i + 1) * int(100 / num_classes) / 100))
            class_range.append([start, end])

        classes = []
        for i in duration:
            for q in range(len(class_range)):
                if class_range[q][0] <= i < class_range[q][1]:
                    classes.append(q)
                elif q == len(class_range) - 1 and i == class_range[len(class_range) - 1][1]:
                    # Special case for the max value.
                    classes.append(3)
        return classes

    @staticmethod
    def build_coordinates(pf, start_x, end_x):
        pf['start'] = [(x, y) for x, y in zip(pf[start_x], pf['start_y'])]
        pf['end'] = [(x, y) for x, y in zip(pf[end_x], pf['end_y'])]
        return pf

    def classify(self, pf, classifier, metric, args, inplace=False):
        for i in range(len(self.segments)):
            pf.loc[pf['segment_index'] == i, 'class'] = classifier(
                pf[pf['segment_index'] == i][metric], *args)
        if inplace:
            self.pf = pf
        else:
            return pf

    def performance_spectrum(self, segments, x_max, segment_height=20):
        """
        Input: segments: Array with defined start and end name of all the segments to be included. x_max: maximum x
        value to be considered when calculating the performance spectrum. classifier: function that will be called with
        as first argument the column of the performance spectrum named 'metric' and with any extra arguments contained
        in list 'args'. segment_height: height of each segment to be plotted.
        """
        self.x_max = x_max
        self.segments = segments
        self.y_s = [[y, y - segment_height] for y in range(segment_height * len(segments), 0, -segment_height)]
        segment_start = [[], []]
        segment_end = [[], []]
        duration = []
        segment_names = []
        segment_index = []
        resource = []
        trace_index = []
        for i in range(len(self.log)):
            def_resource = np.nan
            for n in range(len(self.log[i]) - 1):
                segment = (self.log[i][n][self.event_id], self.log[i][n + 1][self.event_id])
                if segment in segments:
                    start = (self.log[i][n][self.timestamp_id] - self.first).days
                    end = (self.log[i][n + 1][self.timestamp_id] - self.first).days
                    if end <= x_max:
                        duration.append(end - start)
                        x = [start, end]
                        y = self.y_s[segments.index(segment)]
                        segment_start[0].append(x[0])
                        segment_start[1].append(y[0])
                        segment_end[0].append(x[1])
                        segment_end[1].append(y[1])
                        segment_names.append(segment)
                        segment_index.append(segments.index(segment))
                        trace_index.append(i)
                        if self.resource_id:
                            if self.resource_id in self.log[i][n]:
                                def_resource = self.log[i][n][self.resource_id]
                            resource.append(def_resource)

        self.pf['resource'] = resource
        self.pf['start_time'] = segment_start[0]
        self.pf['start_y'] = segment_start[1]
        self.pf['end_time'] = segment_end[0]
        self.pf['end_y'] = segment_end[1]
        self.pf['duration'] = duration
        self.pf['segment_name'] = segment_names
        self.pf['segment_index'] = segment_index
        self.pf['case_id'] = trace_index

    def plot_performance_spectrum(self, class_colors, ax, classifier=None, metric='', args=None, mask=None,
                                  start='start_time',
                                  end='end_time', order=None, compare='global', show_classes=None, vis_mask=False):
        """
        Input: class_colors: list with rgba tuples, there should be a color for each class. ax: A Matplotlib axis
        object. mask: any Pandas mask on the Performance Spectrum Data Frame to be considered before plotting.
        Output: The Matplotlib axis object containing the plotted Performance Spectrum.
        """
        if args is None:
            args = []
        pf = self.pf.copy()
        if vis_mask:
            pf.loc[mask, 'class'] = 1
            pf.loc[~mask, 'class'] = 0
        else:
            if compare == 'global':
                if classifier is not None:
                    pf = self.classify(pf, classifier, metric, args)
                if mask is not None:
                    pf = pf[mask]
            elif compare == 'local':
                if mask is not None:
                    pf = pf[mask]
                if classifier is not None:
                    pf = self.classify(pf, classifier, metric, args)
        if show_classes is not None:
            pf = pf[pf['class'].isin(show_classes)]
        pf = self.build_coordinates(pf, start, end)
        plotting_order = range(len(class_colors)) if order == 'reversed' else reversed(range(len(class_colors)))
        for i in plotting_order:
            lines = [[start, end] for start, end in
                     zip(pf[pf['class'] == i]['start'], pf[pf['class'] == i]['end'])]
            ax.add_collection(
                mc.LineCollection(lines, colors=[class_colors[i] for c in range(len(lines))], alpha=0.25))
        ax.autoscale()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        for i, y in enumerate(self.y_s):
            if type(self.segments[i]) == list:
                text_str = f'{self.segments[i][0]} \n{self.segments[i][1]}'
            else:
                text_str = self.segments[i]
            ax.add_collection(plt.hlines(y, -1*(max(pf['end_time'])/3), max(pf['end_time']), alpha=0.25))
            ax.text(-1*(max(pf['end_time'])/3), y[1] + (y[0]-y[1])/1.5, text_str,
                    fontsize=12,
                    verticalalignment='top', bbox=props)
