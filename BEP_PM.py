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
    def batch_classifier(df, k_min=10, gamma=0):
        """
        Input: df: Dataframe containing the start and end time of each line/(sub-)trace per segment
        k_min: number of subsequent lines/(sub-)traces that must fulfill the batching constraints to be considered a batch 
        gamma: The distance allowed between subsequent lines/(sub-)traces to be considered a batch.
        Output: List of binary class values indicating if a line/(sub-)trace belongs to a batch
        """
        batches = []
        temp_batch = []

        df_sorted = df.sort_values(by=['start_time', 'end_time'], axis=0)
        observations = df_sorted.reset_index()

        temp_batch.append(0)
        for j in range(1, len(observations)):
            if observations['end_time'][j - 1] <= observations['end_time'][j] <= gamma + observations['end_time'][
                j - 1] and observations['start_time'][j] >= observations['start_time'][j - 1]:
                temp_batch.append(j)
                if j == len(observations) - 1 and len(temp_batch) >= k_min:
                    batches.append(temp_batch)
            else:
                if len(temp_batch) >= k_min:
                    batches.append(temp_batch)
                temp_batch = []
                temp_batch.append(j)

        classes = []
        for j in range(len(observations)):
            is_classified = False
            for i in range(len(batches)):
                if j in batches[i]:
                    #                 classes.append(i)
                    classes.append(1)
                    is_classified = True
            if not is_classified:
                #             classes.append(np.nan)
                classes.append(0)
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

    def build_coordinates(self, start_x, end_x):
        self.pf['start'] = [(x, y) for x, y in zip(start_x, self.pf['start_y'])]
        self.pf['end'] = [(x, y) for x, y in zip(end_x, self.pf['end_y'])]

    def classify(self, classifier, metric, args):
        for i in range(len(self.segments)):
            self.pf.loc[self.pf['segment_index'] == i, 'class'] = classifier(
                self.pf[self.pf['segment_index'] == i][metric], *args)

    def performance_spectrum(self, segments, x_max, classifier, metric, args, segment_height=20):
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

        self.build_coordinates(self.pf['start_time'], self.pf['end_time'])
        self.classify(classifier, metric, args)

    def plot_performance_spectrum(self, class_colors, ax, mask=None):
        """
        Input: class_colors: list with rgba tuples, there should be a color for each class. ax: A Matplotlib axis
        object. mask: any Pandas mask on the Performance Spectrum Data Frame to be considered before plotting.
        Output: The Matplotlib axis object containing the plotted Performance Spectrum.
        """
        if mask is not None:
            pf = self.pf[mask].copy()
        else:
            pf = self.pf.copy()

        for i in range(len(class_colors), -1, -1):
            lines = [[start, end] for start, end in
                     zip(pf[pf['class'] == i]['start'], pf[pf['class'] == i]['end'])]
            ax.add_collection(
                mc.LineCollection(lines, colors=[class_colors[i] for c in range(len(lines))], alpha=0.25))
        ax.autoscale()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        for i, y in enumerate(self.y_s):
            text_str = f'{self.segments[i][0]} \n{self.segments[i][1]}'
            ax.text(0.05, y[0] / (abs(ax.get_ylim()[0]) + abs(ax.get_ylim()[1])), text_str, transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top', bbox=props)
            ax.add_collection(plt.hlines(y, 0, max(pf['end_time']), alpha=0.25))
