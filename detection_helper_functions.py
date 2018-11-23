import spikeextractors as se
import numpy as np
from scipy.signal import find_peaks
import collections
Spike = collections.namedtuple('Spike', 'channel time amp')


def getSortedDetectionInfo(sorting):
    '''Returns all detection times and units for a given sorting extractor
       Parameters
       ----------
       sorting: SortingExtractor
           The sorting extractor for the given sorted results, used to get unit ids and firing times.
       Returns
       ----------
       firing_times: array_like
           1D numpy array of firing times in sorted order (earliest to latest in recording)
       firing_neurons: array_like
           1D numpy array of corresponding firing units in same sorted order as firing times
    '''
    firing_neurons = []
    firing_times = []
    for unit_id in sorting.getUnitIds():
        spike_train = sorting.getUnitSpikeTrain(unit_id)
        for spike_time in spike_train:
            firing_neurons.append(unit_id)
            firing_times.append(spike_time)
    firing_neurons = np.asarray(firing_neurons)
    firing_times = np.asarray(firing_times)
    firing_neurons = firing_neurons[np.argsort(firing_times)]
    firing_times = np.sort(firing_times)
    return firing_times, firing_neurons

def runThresholdDetection(channel_ids, start_frame, end_frame, threshold, refractory_period, duplicate_radius, dist_matrix, recording):
    '''A basic threshold crossing detection algorithm
       Parameters
       ----------
       channel_ids: array_like
           1D array or list of all channel_ids in recording
       start_frame: int
           Start frame of detection
       end_frame: int
           End frame of detection
       threshold: float
           Min threshold crossing for a spike to be detected (spikes are flipped to be positive, so threshold is positive)
       refractory_period: int
           Num frames after a detection before a new detection can be registered (helps filter out duplicates also)
       duplicate_radius: float
           Dist (in microns) between channels for detected spikes to be considered duplicate detections with close detection times.
       dist_matrix: 2D array
           Matrix of dists (in microns) between channels. Channels x Channels dimesion
       recording: RecordingExtractor
           RecordingExtractor used to get traces from the recording
       Returns
       ----------
       detected_firing_times: array_like
           1D numpy array of detected firing times in sorted order (earliest to latest in recording)
       detected_channels: array_like
           1D numpy array of corresponding detected channels in same sorted order as detected firing times
    '''
    channel_spike_times = []
    detected_channels = []
    channel_peak_amps = []
    for channel_id in channel_ids:
        peaks, peak_amps = find_peaks(-recording.getTraces(channel_ids=[channel_id], start_frame=start_frame, end_frame=end_frame)[0], 
                                      height=threshold, 
                                      distance=refractory_period, 
                                      prominence=None, 
                                      width=None, 
                                      wlen=None, 
                                      rel_height=0.5)
        detected_channels.append(np.asarray([channel_id]*peaks.shape[0]))
        channel_spike_times.append(peaks)
        channel_peak_amps.append(peak_amps['peak_heights'])
        
    detected_channels = np.concatenate(np.asarray(detected_channels))
    channel_spike_times = np.concatenate(np.asarray(channel_spike_times))
    channel_peak_amps = np.concatenate(np.asarray(channel_peak_amps))

    inds = np.argsort(channel_spike_times)
    channel_spike_times = channel_spike_times[inds]
    detected_channels = detected_channels[inds]
    channel_peak_amps = channel_peak_amps[inds]
    
    stored_spikes = []
    final_spikes = []
    for i, spike_time in enumerate(channel_spike_times):
        detected_channel = detected_channels[i]
        peak_amp = channel_peak_amps[i]
        if(len(stored_spikes) != 0):
            while(spike_time - stored_spikes[0][1] > refractory_period):
                stored_spikes, final_spikes = processStoredSpikes(stored_spikes, dist_matrix, duplicate_radius, final_spikes)
                if(len(stored_spikes) == 0):
                    break
        stored_spikes.append(Spike(detected_channel, spike_time, peak_amp))
    
    detected_firing_times = []
    detected_channels = []
    for spike in final_spikes:
        detected_firing_times.append(spike.time)
        detected_channels.append(spike.channel)
        
    return np.asarray(detected_firing_times), np.asarray(detected_channels)

def processStoredSpikes(stored_spikes, dist_matrix, duplicate_radius, final_spikes):
    '''Helper function for threshold detection method. Used to filter duplicate events from stored_spikes and return true events
       Parameters
       ----------
       stored_spikes: list
           List of spikes detected on each channel (duplicates included)
       dist_matrix: 2D array
           Matrix of dists (in microns) between channels. Channels x Channels dimesion
       duplicate_radius: float
           Dist (in microns) between channels for detected spikes to be considered duplicate detections with close detection times.
       final_spikes: list
           Empty list to store largest amp spikes (true events)
       Returns
       ----------
       new_stored_spikes: list
           List of spikes detected on each channel (duplicates removed)
       final_spikes: list
           Filled list returned containing largest amp spikes (true events)
    '''
    new_stored_spikes = []
    first_spike = stored_spikes[0]
    curr_max_amp = first_spike.amp
    curr_max_channel = first_spike.channel
    curr_max_spike = first_spike
    for curr_spike in stored_spikes[1:]:
        if(dist_matrix[first_spike.channel][curr_spike.channel] <= duplicate_radius):
            if(curr_max_amp < curr_spike.amp):
                curr_max_amp = curr_spike.amp
                curr_max_channel = curr_spike.channel
                curr_max_spike = curr_spike
        else:
            new_stored_spikes.append(curr_spike)
    final_spikes.append(curr_max_spike)
    return new_stored_spikes, final_spikes

def evaluateDetection(detected_firing_times, detected_channels, firing_times, firing_neurons, channel_positions, 
                      neuron_locs, max_neuron_channel_dist, jitter=10):
    '''Function to return results of detection algorithm. Matches detected events to ground truth and returns all matched events,
       all unmatched detections, and all unmatched ground truth firings (true positives, false positives, and false negatives,
       respectively).
       Parameters
       ----------
       detected_firing_times: array_like
           1D numpy array of detected firing times in sorted order (earliest to latest in recording)
       detected_channels: array_like
           1D numpy array of corresponding detected channels in same sorted order as detected firing times
       firing_times: array_like
           1D numpy array of firing times in sorted order (earliest to latest in recording)
       firing_neurons: array_like
           1D numpy array of corresponding firing units in same sorted order as firing times
       channel_positions: 2D array
           2D numpy array of x, y, z positions for each channel
       neuron_locs: 2D array
           2D numpy array of x, y, z positions for each unit (either true neuron location)
       max_neuron_channel_dist: float
           Dist (in microns) from channels to unit locations for them to be considered candidates for the spike
       jitter: int
           The frames before or after a detection for the detection to be matched to a true firing time
       Returns
       ----------
       matched_events: list
           List of tuples (Detected frame, matched ground truth frame, detected channel) in chronological order (TP)
       unmatched_detections: list
           List of tuples (Detected frame, detected channel) in chronological order (FP)
       unmatched_firings: list
           List of tuples (Firing time, firing neuron) in chronological order (FN)
    '''
    unmatched_detections = []
    matched_events = []
    copy_firing_times = np.copy(firing_times)
    copy_firing_neurons = np.copy(firing_neurons)
    
    for i, detected_time in enumerate(detected_firing_times):
        detected_channel = detected_channels[i]
        candidate_events = []
        times_to_be_deleted = []
        neurons_to_be_deleted = []
        for j, firing_time in enumerate(copy_firing_times):
            firing_neuron = copy_firing_neurons[j]
            if(detected_time >= firing_time - jitter):
                if(detected_time <= firing_time + jitter):
                    dist_to_neuron = getDist(channel_positions[detected_channel], neuron_locs[firing_neuron])
                    if(dist_to_neuron < max_neuron_channel_dist):
                        candidate_events.append((j, dist_to_neuron, detected_channel, firing_time))
            else:
                if(len(candidate_events) == 0):
                    unmatched_detections.append((detected_time, detected_channel))
                    break
                if(len(candidate_events) == 1):
                    idx, _, detected_channel, firing_time = candidate_events[0]
                    matched_events.append((detected_time,firing_time, detected_channel))
                    copy_firing_times = np.delete(copy_firing_times, idx)
                    copy_firing_neurons = np.delete(copy_firing_neurons, idx)
                    break
                else:
                    first_event = candidate_events[0]
                    closest_event = first_event
                    for curr_event in candidate_events[1:]:
                        if(curr_event[1] < closest_event[1]):
                            closest_event = curr_event
                    matched_events.append((detected_time,closest_event[3], closest_event[2]))
                    copy_firing_times = np.delete(copy_firing_times, closest_event[0])
                    copy_firing_neurons = np.delete(copy_firing_neurons, closest_event[0])
                    break
    unmatched_firings = list(zip(copy_firing_times, copy_firing_neurons))                
    return matched_events, unmatched_detections, unmatched_firings
    
def getNClosestPositions(N, point, positions):
    '''Returns N closest position ids to the given point from the positions array
       Parameters
       ----------
       point: array_like
           1D numpy array of x, y, z position
       positions: 2D array
           2D numpy array of x, y, z positions
       Returns
       ----------
       N_closest_position_ids: array_like
           1D array of closest ids to point in positions
       N_closest_dists: array_like
           1D array of corresponding dists of the ids in positions to the point
    '''
    dists = np.sqrt(np.sum((positions-point)**2, axis=1))
    closest_position_ids = np.argsort(dists)
    sorted_dists = np.sort(dists)
    N_closest_position_ids = closest_position_ids[:N]
    N_closest_dists = sorted_dists[:N]
    return N_closest_position_ids, N_closest_dists

    
def getDist(point1, point2):
    '''Returns euclidean distance between two points
       Parameters
       ----------
       point1: array_like
           1D numpy array of x, y, z position
       point2: array_like
           1D numpy array of x, y, z position
       Returns
       ----------
       dist: float
           Euclidean distance between point1 and point2
    '''
    dist = np.sqrt(np.sum((point1-point2)**2))
    return dist