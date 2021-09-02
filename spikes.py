import numpy as np
import os

class Spikes():

    def __init__(self, clusters, type, home_dir=None):
        self.clusters = clusters
        self.type = type
        if home_dir is None:
            home_dir = os.getcwd()
        self.home_dir = home_dir


class KsSpikes(Spikes):
    def __init__(self, home_dir=None, **kwargs):
        self.params = self.default_params()
        for i in kwargs.keys():
            if i in self.params:
                self.params[i] = kwargs[i]
        if home_dir is not None:
            orig_dir = os.getcwd()
            os.chdir(home_dir)
        self.clusters, self.cluster_ints, self.cluster_groups = self._find_spikes()
        Spikes.__init__(self, self.clusters, 'Kilosort', home_dir)
        if home_dir is not None:
            os.chdir(orig_dir)

    def default_params(self):
        params = {
            'spike_times_fn':'spike_times.npy',
            'spike_clusters_fn':'spike_clusters.npy',
            'cluster_group_fn':'cluster_group.tsv'
        }
        return params

    def _find_spikes(self):
        params = self.default_params()
        spike_file, cluster_file, cluster_groups = params['spike_times_fn'], params['spike_clusters_fn'], params['cluster_group_fn']
        spike_times = np.load(spike_file)
        spike_clusters = np.load(cluster_file)
        cluster_groups = open('cluster_group.tsv', 'r').read()
        spike_clusters_set = list(set(spike_clusters.flatten()))
        cluster_groups = cluster_groups.split('\n')[1:-1]
        clusters = [spike_times[spike_clusters==i] for i in list(set(np.concatenate(spike_clusters)))]
        clusters = [i.astype(np.int64) for i in clusters]
        #good_clusts_ints = [index for index, i in enumerate(cluster_groups[1:-1]) if i.split('\t')[1] == 'good']
        #good_clust_index = [index for index, i in enumerate(spike_clusters_set) if i in good_clusts_ints]
        clusters = np.array(clusters)
        return clusters, spike_clusters_set, cluster_groups
    
    def get_good_clusters(self):
        good_clusts_ints = [index for index, i in enumerate(self.cluster_groups) if i.split('\t')[1] == 'good']
        good_clust_index = [index for index, i in enumerate(self.cluster_ints) if i in good_clusts_ints]
        good_clusters = self.clusters[good_clust_index]
        return good_clusters