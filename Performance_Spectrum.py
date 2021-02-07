import matplotlib.pyplot as plt
import pickle
import datetime
from matplotlib import collections as mc
import numpy as np
import pandas as pd
from pm4py.objects.log import log


def load_log(name):
    return pickle.load(open(name + '.dat', 'rb'))


class EventLog:
    def __init__(self, name, eventlog):
        self.timestamp_id = 'time:timestamp'
        self.event_id = 'concept:name'

        self.segment_count = {}
        self.log = eventlog
        print('> Cleaning time zone information, adjust to 0.')
        self.clean_timezone()

        # Set the begin and end date.
        self.first = min(
            [min(self.log[i][k]['time:timestamp'] for k in range(len(self.log[i]))) for i in range(len(self.log))])
        self.last = max(
            [max(self.log[i][k]['time:timestamp'] for k in range(len(self.log[i]))) for i in range(len(self.log))])

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
        print(f'> Found dates in event log ranging from {self.first} to {self.last}.')
        print('> Done, continue with load_frame.')

    def load_frame(self, maxdate=None):
        if maxdate:
            matches_date = [i for i in range(len(self.log)) if self.log[i][0]['time:timestamp'] < maxdate]
        else:
            matches_date = range(len(self.log))
        print('> Creating Pandas DataFrame from event log, this might take a while...')
        pf = pd.concat([pd.DataFrame(self.log[i]) for i in matches_date], sort=False)
        pf.loc[pf.index == 0, 'case_id'] = [i for i in range((pf.index == 0).sum())]
        pf['case_id'] = pf['case_id'].fillna(method='ffill')

        self.pf = pf.sort_values('case_id').copy()
        self.pf['org:resource'] = self.pf['org:resource'].fillna(method='ffill')
        self.pf.reset_index(drop=True, inplace=True)
        print('> Done, continue with inspection of pf and create_structure')

    def create_structure(self, concept_names_mask=None):
        if concept_names_mask is not None:
            df = self.pf[concept_names_mask].sort_values(['case_id', 'time:timestamp', 'concept:name'])
        else:
            df = self.pf.sort_values(['case_id', 'time:timestamp', 'concept:name'])
        print('> Shifting log to generate case transitions.')
        t = pd.concat([df, df.shift(-1)], axis=1)
        t.columns = [str(i) + str(k // (len(df.columns))) for k, i in enumerate(t.columns)]
        t = t[t['case_id0'] == t['case_id1']]

        t['segment_name'] = t['concept:name0'] + ' ' + t['lifecycle:transition0'] + ' - ' + t['concept:name1'] + ' ' + \
                            t['lifecycle:transition1']
        t.drop(['concept:name0', 'lifecycle:transition0', 'lifecycle:transition1', 'case_id1', 'concept:name1'], axis=1,
               inplace=True)
        t.columns = ['start_' + i.split('0')[0] if '0' in i else ('end_' + i.split('1')[0] if '1' in i else i) for i in
                     t.columns]

        self.segments = t['segment_name'].unique()
        self.pf = t.copy()
        print('> Adjusting time-grain to seconds.')
        self.pf['end_time:timestamp'] = self.pf['end_time:timestamp'] - self.first
        self.pf['end_time:timestamp'] = self.pf['end_time:timestamp'].dt.days * 24 + round(
            self.pf['end_time:timestamp'].dt.seconds / 3600)

        self.pf['start_time:timestamp'] = self.pf['start_time:timestamp'] - self.first
        self.pf['start_time:timestamp'] = self.pf['start_time:timestamp'].dt.days * 24 + round(
            self.pf['start_time:timestamp'].dt.seconds / 3600)

        self.pf['duration'] = self.pf['end_time:timestamp'] - self.pf['start_time:timestamp']

        self.pf['segment_index'] = self.pf['segment_name'].apply(lambda x: list(self.segments).index(x))
        print('> Done, continue with remove_meta to remove columns from pf and to correct naming.')

    def remove_meta(self, to_keep=None):
        if to_keep is None:
            to_keep = []
        columns = ['start_case_id', 'start_org:resource', 'end_org:resource', 'start_time:timestamp',
                   'end_time:timestamp', 'segment_name', 'duration', 'segment_index']
        columns.extend(to_keep)
        self.pf = self.pf[columns]
        renamed = ['case_id', 'start_org:resource', 'end_org:resource', 'start_time', 'end_time', 'segment_name',
                   'duration', 'segment_index']
        renamed.extend(to_keep)
        self.pf.columns = renamed

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
            pickle.dump(self, f)

    def segment_counts(self):
        return dict(self.pf['segment_name'].value_counts())

    def filter_segments(self, cutoff=0.25, compare_to='previous'):
        """
        Input: a cutoff value for deciding which segments to include as a ratio of minimum amount of occurrence
        compared to the occurrence of either the maximum or previous segment.
        Output: Frequently occurring segments according to cutoff value.
        """
        sorted_counts = self.segment_counts()
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
        return filtered_segments

    def infer_start_times(self):
        t = self.pf.copy()
        t['start'] = [i[0] for i in self.pf['segment_name'].str.split(' - ')]
        t['end'] = [i[1] for i in self.pf['segment_name'].str.split(' - ')]
        s = t[['case_id', 'start_org:resource', 'start_time', 'start']]
        s.columns = ['case_id', 'org:resource', 'end_time', 'segment_name']
        e = t[['case_id', 'end_org:resource', 'end_time', 'end']]
        e.columns = ['case_id', 'org:resource', 'end_time', 'segment_name']
        t = pd.concat([e, s], axis=0).sort_values(['org:resource', 'end_time']).reset_index(drop=True)
        t['start_time'] = t.shift()[t.shift()['org:resource'] == t['org:resource']]['end_time']
        t.sort_values('case_id', inplace=True)
        t['duration'] = t['end_time'] - t['start_time']
        return t[t['duration'] > 0]


class Spectrum:
    def __init__(self, segments, pf):
        self.segments = segments
        self.pf = pf[pf['segment_name'].isin(self.segments)].copy()
        self.pf['segment_index'] = self.pf['segment_name'].apply(lambda x: list(self.segments).index(x))

    @staticmethod
    def classify_hist(duration, num_classes):
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

    def classify(self, pf, classifier, metric, args, inplace=False):
        for i in range(len(self.segments)):
            pf.loc[pf['segment_index'] == i, 'class'] = classifier(
                pf[pf['segment_index'] == i][metric], *args)
        if inplace:
            self.pf = pf
        else:
            return pf

    def build_coordinates(self, pf, start_x, end_x):
        pf['start_y'] = (200 // len(self.segments)) * (len(self.segments) - pf['segment_index'])
        pf['end_y'] = pf['start_y'] - (200 // len(self.segments))
        self.y_s = [[y, y - (200 // len(self.segments))] for y in
                    range((200 // len(self.segments)) * len(self.segments), 0, -(200 // len(self.segments)))]
        pf['start'] = [(x, y) for x, y in zip(pf[start_x], pf['start_y'])]
        pf['end'] = [(x, y) for x, y in zip(pf[end_x], pf['end_y'])]
        return pf

    def plot_performance_spectrum(self, class_colors, ax, classifier=None, metric='', args=None, mask=None,
                                  start='start_time',
                                  end='end_time', order=None, compare='global', show_classes=None, vis_mask=False,
                                  label_offset=0, exclude=None):
        """
        Input: class_colors: list with rgba tuples, there should be a color for each class. ax: A Matplotlib axis
        object. mask: any Pandas mask on the Performance Spectrum Data Frame to be considered before plotting.
        Output: The Matplotlib axis object containing the plotted Performance Spectrum.
        """
        if args is None:
            args = []

        if exclude is not None:
            pf = self.pf[~exclude].copy()
        else:
            pf = self.pf.copy()

        if classifier is None and not vis_mask:
            print(
                '> No classifier defined, vis_mask=False so no coloring based on mask, coloring eveything in first color!')
            pf['class'] = 0

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
            ax.add_collection(plt.hlines(y, label_offset, max(pf['end_time']), alpha=0.25))
            ax.text(label_offset, y[1] + (y[0] - y[1]) / 1.5, text_str,
                    fontsize=12,
                    verticalalignment='top', bbox=props)
