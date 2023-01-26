import phylodeep.sumstats as sumstats
from collections import Counter
import pandas as pd
import numpy as np
import sys
from ete3 import Tree

import warnings
warnings.filterwarnings("ignore")

# set high recursion limit
sys.setrecursionlimit(10000)

###Simulation of the trees

# initiate attributes of nodes
# reasons why a branch stops (transmission, removed/sampled/unsampled tips before the end of simulations)
STOP_REASON = 'stop_reason'


# initiate attributes of nodes
# reasons why a branch stops (transmission, removed/sampled/unsampled tips before the end of simulations)
STOP_REASON = 'stop_reason'
STOP_UNKNOWN = 0
STOP_TRANSMISSION = 1
STOP_REMOVAL_WOS = 2
STOP_SAMPLING = 3
STOP_TIME = 4

# infectious types: their meaning (eg normal/superspreading) depends on the values of parameters
I_1 = 1
I_2 = 2
I_T = 'i_t'

HISTORY = 'history'
SAMPLING = 'sampling'
TRANSMISSION = 'transmission'
DIST_TO_START = 'DIST_TO_START'
PROCESSED = 'processed'

# import file with a table of parameter values
design = pd.read_csv(r'C:\Users\LORENA.LAPTOP-LGLJM15L\Documents\Cours ENS\2022-2023\deep_learning\projet\parameters_tree.csv')

nb_samples = len(design)

# should not be constraining (eg Infinity), depends on your experiment
maxTime = 1000

def simulate_bdss_tree_gillespie(tr_r11, tr_r12, tr_r21, tr_r22, removal_r, sampling_p, max_s, max_t,
                                 fraction_1):
    """
    Simulates the tree evolution with heterogeneous hosts (of type t1 and t2) based on the given transmission rates,
     removal rate, sampling probabilities and number of tips
    :param tr_r11: float of transmission rate from t1 to t1 type spreader
    :param tr_r12: float of transmission rate from t1 to t2 type spreader
    :param tr_r21: float of transmission rate from t2 to t1 type spreader
    :param tr_r22: float of transmission rate from t2 to t2 type spreader
    :param removal_r: float of removal rate of both t1 and t2 type spreaders
    :param sampling_p: float, between 0 and 1, probability for removed t1 tip/spreader to be immediately sampled
    :param max_s: int, maximum number of sampled leaves in a tree
    :param max_t: float, maximum time from root simulation
    :param fraction_1: float, between 0 and 1, fraction of type 1 spreaders
    :return: the simulated tree (ete3.Tree).
    """
    right_size = 0
    trial = 0

    def initialize(init_type):
        if init_type == 1:
            metrics['number_inf1_leaves'] = 1
            leaves_dict['inf1_leaves'] = root.get_leaves()

            metrics['total_inf1'] = 1
        else:
            metrics['number_inf2_leaves'] = 1
            leaves_dict['inf2_leaves'] = root.get_leaves()

            metrics['total_inf2'] = 1
        return None

    def update_rates(rates, metrics_dc):
        """
        updates rates dictionary
        :param rates: dict, all rate values at previous step
        :param metrics_dc: dict, counts of different individuals, list(s) of different branch types
        :return:
        """
        rates['transmission_rate_1_1_i'] = tr_r11 * metrics_dc['number_inf1_leaves']
        rates['transmission_rate_1_2_i'] = tr_r12 * metrics_dc['number_inf1_leaves']
        rates['transmission_rate_2_1_i'] = tr_r21 * metrics_dc['number_inf2_leaves']
        rates['transmission_rate_2_2_i'] = tr_r22 * metrics_dc['number_inf2_leaves']

        rates['removal_rate_t1_i'] = removal_r * metrics_dc['number_inf1_leaves']
        rates['removal_rate_t2_i'] = removal_r * metrics_dc['number_inf2_leaves']

        return None

    def transmission(t_donor, t_recipient):
        if t_donor == 1:
            # which leaf will be affected by the event?
            nb_which_leaf = np.random.randint(0, metrics['number_inf1_leaves'])
            which_leaf = leaves_dict['inf1_leaves'][nb_which_leaf]
            del leaves_dict['inf1_leaves'][nb_which_leaf]
        else:
            # which leaf will be affected by the event?
            nb_which_leaf = np.random.randint(0, metrics['number_inf2_leaves'])
            which_leaf = leaves_dict['inf2_leaves'][nb_which_leaf]
            del leaves_dict['inf2_leaves'][nb_which_leaf]

        # which_leaf becomes an internal node
        which_leaf.dist = abs(time - which_leaf.DIST_TO_START)
        which_leaf.add_feature(DIST_TO_START, time)
        which_leaf.add_feature(STOP_REASON, STOP_TRANSMISSION)

        # which_leaf gives birth to recipent and donor
        recipient, donor = which_leaf.add_child(dist=0), which_leaf.add_child(dist=0)
        recipient.add_feature(DIST_TO_START, which_leaf.DIST_TO_START)
        donor.add_feature(DIST_TO_START, which_leaf.DIST_TO_START)

        # add recipient to its lists, add it its attributes:
        if t_recipient == 1:
            recipient.add_feature(I_T, I_1)
            leaves_dict['inf1_leaves'].append(recipient)
            metrics['total_inf1'] += 1
            metrics['number_inf1_leaves'] += 1
        else:
            recipient.add_feature(I_T, I_2)
            leaves_dict['inf2_leaves'].append(recipient)
            metrics['total_inf2'] += 1
            metrics['number_inf2_leaves'] += 1

        # add donor to its lists, add it its attributes:
        if t_donor == 1:
            donor.add_feature(I_T, I_1)
            leaves_dict['inf1_leaves'].append(donor)
        else:
            donor.add_feature(I_T, I_2)
            leaves_dict['inf2_leaves'].append(donor)

        metrics['total_branches'] += 1
        return None

    def removal(removed_type):
        """
        updates the tree, the metrics and leaves_dict following a removal event
        :param removed_type: int, either 1: a branch of type 1 undergoes removal; 2: a branch of type 2 undergoes
        removal
        :return:
        """
        # which leaf is removed?: "which_leaf"
        if removed_type == 1:
            nb_which_leaf = np.random.randint(0, metrics['number_inf1_leaves'])
            which_leaf = leaves_dict['inf1_leaves'][nb_which_leaf]
            del leaves_dict['inf1_leaves'][nb_which_leaf]
            metrics['number_inf1_leaves'] -= 1
        else:
            nb_which_leaf = np.random.randint(0, metrics['number_inf2_leaves'])
            which_leaf = leaves_dict['inf2_leaves'][nb_which_leaf]
            leaves_dict['inf2_leaves'][nb_which_leaf]
            metrics['number_inf2_leaves'] -= 1

        # which_leaf becomes a tip
        which_leaf.dist = abs(time - which_leaf.DIST_TO_START)
        which_leaf.add_feature(DIST_TO_START, time)
        which_leaf.add_feature(PROCESSED, True)

        # was which_leaf sampled?
        if np.random.rand() < sampling_p:
            metrics['number_sampled'] += 1
            which_leaf.add_feature(STOP_REASON, STOP_SAMPLING)
            metrics['total_removed'] += 1
            if removed_type == 1:
                metrics['sampled_inf1'] += 1
            else:
                metrics['sampled_inf2'] += 1
        else:
            which_leaf.add_feature(STOP_REASON, STOP_REMOVAL_WOS)
            metrics['total_removed'] += 1
        return None

    # up to 100 times retrial of simulation until reaching correct size
    while right_size == 0 and trial < 100:

        root = Tree(dist=0)
        root.add_feature(DIST_TO_START, 0)

        # initiate the time of simulation
        time = 0

        # INITIATE: metrics counting leaves and branches of different types, leaves_dict storing all leaves alive, and
        # rates_i with all rates at given time, for Gillespie algorithm
        metrics = {'total_branches': 1, 'total_removed': 0, 'number_sampled': 0, 'total_inf1': 0, 'total_inf2': 0,
                   'sampled_inf1': 0, 'sampled_inf2': 0, 'number_inf1_leaves': 0, 'number_inf2_leaves': 0}
        leaves_dict = {'inf1_leaves': [], 'inf2_leaves': []}

        rates_i = {'removal_rate_t1_i': 0, 'removal_rate_t2_i': 0, 'transmission_rate_1_1_i': 0,
                   'transmission_rate_1_2_i': 0, 'transmission_rate_2_1_i': 0, 'transmission_rate_2_2_i': 0}

        # INITIATE: of which type is the first branch?
        # first individual will be of type 1 with probability frac_1 (frequence of type 1 at equilibrium)
        if np.random.rand() < fraction_1:
            initialize(1)

        else:
            initialize(2)

        # simulate while [1] the epidemics do not go extinct, [2] given number of patients were not sampled,
        # [3] maximum time of simulation was not reached
        while (metrics['number_inf1_leaves'] + metrics['number_inf2_leaves']) > 0 \
                and (metrics['number_sampled'] < max_s) and (time < max_t):
            # first we need to re-calculate the rates and take its sum
            update_rates(rates_i, metrics)
            sum_rates_i = sum(rates_i.values())

            # when does the next event take place?
            time_to_next = np.random.exponential(1 / sum_rates_i, 1)[0]
            time = time + time_to_next

            # which event will happen
            random_event = np.random.uniform(0, 1, 1) * sum_rates_i

            if random_event < rates_i['transmission_rate_1_1_i']:
                # there will be a transmission event from t1 to t1 type spreader
                transmission(1, 1)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i']):
                # there will be a transmission event from t1 to t2 type spreader
                transmission(1, 2)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i'] +
                                 rates_i['transmission_rate_2_1_i']):
                # there will be a transmission event from t2 to t1 type spreader
                transmission(2, 1)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i'] +
                                 rates_i['transmission_rate_2_1_i'] + rates_i['transmission_rate_2_2_i']):
                transmission(2, 2)

            elif random_event < \
                    (sum_rates_i - rates_i['removal_rate_t2_i']):
                # there will be a removal event of t1 type spreader
                removal(removed_type=1)

            else:
                removal(removed_type=2)
                # there will be a removal event of a t2 type spreader

        # tag non-removed tips at the end of simulation
        for leaflet in root.get_leaves():
            if getattr(leaflet, STOP_REASON, False) != 2 and getattr(leaflet, STOP_REASON, False) != 3:
                leaflet.dist = abs(time - leaflet.DIST_TO_START)
                leaflet.add_feature(DIST_TO_START, time)
                leaflet.add_feature(STOP_REASON, STOP_TIME)

        if metrics['number_sampled'] == max_s:
            right_size = 1
        else:
            trial += 1

    # statistics on the simulation
    del metrics['number_inf1_leaves']
    del metrics['number_inf2_leaves']
    vector_count = list(metrics.values())
    vector_count.extend([time, trial])

    return root, vector_count

def _merge_node_with_its_child(nd, child=None, state_feature=STOP_REASON):
    if not child:
        child = nd.get_children()[0]
    nd_hist = getattr(nd, HISTORY, [(getattr(nd, state_feature, ''), 0)])
    nd_hist += [('!', nd.dist - sum(it[1] for it in nd_hist))] \
               + getattr(child, HISTORY, [(getattr(child, state_feature, ''), 0)])
    child.add_features(**{HISTORY: nd_hist})
    child.dist += nd.dist
    if nd.is_root():
        child.up = None
    else:
        parent = nd.up
        parent.remove_child(nd)
        parent.add_child(child)
    return child


def remove_certain_leaves(tre, to_remove=lambda node: False, state_feature=STOP_REASON):
    """
    Removes all the branches leading to naive leaves from the given tree.
    :param tre: the tree of interest (ete3 Tree)
    [(state_1, 0), (state_2, time_of_transition_from_state_1_to_2), ...]. Branch removals will be added as '!'.
    :param to_remove: a method to check if a leaf should be removed.
    :param state_feature: the node feature to store the state
    :return: the tree with naive branches removed (ete3 Tree) or None is all the leaves were naive in the initial tree.
    """

    for nod in tre.traverse("postorder"):
        # If this node has only one child branch
        # it means that the other child branch used to lead to a naive leaf and was removed.
        # We can merge this node with its child
        # (the child was already processed and either is a leaf or has 2 children).
        if len(nod.get_children()) == 1:
            merged_node = _merge_node_with_its_child(nod, state_feature=state_feature)
            if merged_node.is_root():
                tre = merged_node
        elif nod.is_leaf() and to_remove(nod):
            if nod.is_root():
                return None
            nod.up.remove_child(nod)
    return tre

###Processing of the trees


DISTANCE_TO_ROOT = "dist_to_root"

DEPTH = "depth"

HEIGHT = "height"

LADDER = "ladder"

VISITED = "visited"

# all branches of given tree will be rescaled to TARGET_AVG_BL
TARGET_AVG_BL = 1

def add_dist_to_root(tre):
    """
    Add distance to root (dist_to_root) attribute to each node
    :param tre: ete3.Tree, tree on which the dist_to_root should be added
    :return: void, modifies the original tree
    """

    for node in tre.traverse("preorder"):
        if node.is_root():
            node.add_feature("dist_to_root", 0)
        elif node.is_leaf():
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
        else:
            node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
            # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
    return None


def name_tree(tre):
    """
    Names all the tree nodes that are not named, with unique names.
    :param tre: ete3.Tree, the tree to be named
    :return: void, modifies the original tree
    """
    existing_names = Counter((_.name for _ in tre.traverse() if _.name))
    i = 0
    for node in tre.traverse('levelorder'):
        node.name = i
        i += 1
    return None
###Computation of the compact bijective ladderized representation

def encode_into_most_recent(tree_input, sampling_proba):
    """Rescales all trees from tree_file so that mean branch length is 1,
    then encodes them into full tree representation (most recent version)
    :param tree_input: ete3.Tree, that we will represent in the form of a vector
    :param sampling_proba: float, value between 0 and 1, presumed sampling probability value
    :return: pd.Dataframe, encoded rescaled input trees in the form of most recent, last column being
     the rescale factor
    """

    def real_polytomies(tre):
        """
        Replaces internal nodes of zero length with real polytomies.
        :param tre: ete3.Tree, the tree to be modified
        :return: void, modifies the original tree
        """
        for nod in tre.traverse("postorder"):
            if not nod.is_leaf() and not nod.is_root():
                if nod.dist == 0:
                    for child in nod.children:
                        nod.up.add_child(child)
                    nod.up.remove_child(nod)
        return

    def get_not_visited_anc(leaf):
        while getattr(leaf, "visited", 0) >= len(leaf.children)-1:
            leaf = leaf.up
            if leaf is None:
                break
        return leaf

    def get_deepest_not_visited_tip(anc):
        max_dist = -1
        tip = None
        for leaf in anc:
            if leaf.visited == 0:
                distance_leaf = getattr(leaf, "dist_to_root") - getattr(anc, "dist_to_root")
                if distance_leaf > max_dist:
                    max_dist = distance_leaf
                    tip = leaf
        return tip

    def get_dist_to_root(anc):
        dist_to_root = getattr(anc, "dist_to_root")
        return dist_to_root

    def get_dist_to_anc(feuille, anc):
        dist_to_anc = getattr(feuille, "dist_to_root") - getattr(anc, "dist_to_root")
        return dist_to_anc

    def encode(anc):
        leaf = get_deepest_not_visited_tip(anc)
        yield get_dist_to_anc(leaf, anc)
        leaf.visited += 1
        anc = get_not_visited_anc(leaf)

        if anc is None:
            return
        anc.visited += 1
        yield get_dist_to_root(anc)
        for _ in encode(anc):
            yield _

    def complete_coding(encoding, max_length):
        add_vect = np.repeat(0, max_length - len(encoding))
        add_vect = list(add_vect)
        encoding.extend(add_vect)
        return encoding

    def refactor_to_final_shape(result_v, sampling_p, maxl):
        def reshape_coor(max_length):
            tips_coor = np.arange(0, max_length, 2)
            tips_coor = np.insert(tips_coor, -1, max_length + 1)

            int_nodes_coor = np.arange(1, max_length - 1, 2)
            int_nodes_coor = np.insert(int_nodes_coor, 0, max_length)
            int_nodes_coor = np.insert(int_nodes_coor, -1, max_length + 2)

            order_coor = np.append(int_nodes_coor, tips_coor)

            return order_coor

        reshape_coordinates = reshape_coor(maxl)

        # add sampling probability:
        if maxl == 999:
            result_v.loc[:, 1000] = 0
            result_v['1001'] = sampling_p
            result_v['1002'] = sampling_p
        else:
            result_v.loc[:, 400] = 0
            result_v['401'] = sampling_p
            result_v['402'] = sampling_p

        # reorder the columns
        result_v = result_v.iloc[:,reshape_coordinates]

        return result_v

    # local copy of input tree
    tree = tree_input.copy()

    if len(tree) < 200:
        max_len = 399
    else:
        max_len = 999

    # remove the edge above root if there is one
    if len(tree.children) < 2:
        tree = tree.children[0]
        tree.detach()

    # set to real polytomy
    real_polytomies(tree)

    # rescale branch lengths
    rescale_factor = rescale_tree(tree, target_avg_length=TARGET_AVG_BL)

    # set all nodes to non visited:
    for node in tree.traverse():
        setattr(node, "visited", 0)

    name_tree(tree)

    add_dist_to_root(tree)

    tree_embedding = list(encode(tree))

    tree_embedding = complete_coding(tree_embedding, max_len)
    #tree_embedding.append(rescale_factor)

    result = pd.DataFrame(tree_embedding, columns=[0])

    result = result.T
    # refactor to final shape: add sampling probability, put features in order

    result = refactor_to_final_shape(result, sampling_proba, max_len)

    return result, rescale_factor

###Computation of the summary statistics

def rescale_tree(tre, target_avg_length):
    """
    Returns branch length metrics (all branches taken into account and external only)
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param target_avg_length: float, the average branch length to which we want to rescale the tree
    :return: float, resc_factor
    """
    # branch lengths
    dist_all = [node.dist for node in tre.traverse("levelorder")]

    all_bl_mean = np.mean(dist_all)

    resc_factor = all_bl_mean/target_avg_length

    for node in tre.traverse():
        node.dist = node.dist/resc_factor

    return resc_factor


def encode_into_summary_statistics(tree_input, sampling_proba):
    """Rescales all trees from tree_file so that mean branch length is 1,
    then encodes them into summary statistics representation
    :param tree_input: ete3.Tree, on which the summary statistics will be computed
    :param sampling_proba: float, presumed sampling probability for all the trees
    :return: pd.DataFrame, encoded rescaled input trees in the form of summary statistics and float, a rescale factor
    """
    # local copy of input tree
    tree = tree_input.copy()

    # compute the rescale factor
    rescale_factor = rescale_tree(tree, target_avg_length=TARGET_AVG_BL)

    # add accessory attributes
    name_tree(tree)
    max_depth = sumstats.add_depth_and_get_max(tree)
    add_dist_to_root(tree)
    sumstats.add_ladder(tree)
    sumstats.add_height(tree)

    # compute summary statistics based on branch lengths
    summaries = []
    summaries.extend(sumstats.tree_height(tree))
    summaries.extend(sumstats.branches(tree))
    summaries.extend(sumstats.piecewise_branches(tree, summaries[0], summaries[5], summaries[6], summaries[7]))
    summaries.append(sumstats.colless(tree))
    summaries.append(sumstats.sackin(tree))
    summaries.extend(sumstats.wd_ratio_delta_w(tree, max_dep=max_depth))
    summaries.extend(sumstats.max_ladder_il_nodes(tree))
    summaries.extend(sumstats.staircaseness(tree))

    # compute summary statistics based on LTT plot

    ltt_plot_matrix = sumstats.ltt_plot(tree)
    summaries.extend(sumstats.ltt_plot_comput(ltt_plot_matrix))

    # compute LTT plot coordinates

    summaries.extend(sumstats.coordinates_comp(ltt_plot_matrix))

    # compute summary statistics based on transmission chains (order 4):

    summaries.append(len(tree))
    summaries.extend(sumstats.compute_chain_stats(tree, order=4))
    summaries.append(sampling_proba)

    result = pd.DataFrame(summaries, columns=[0])

    result = result.T

    return result, rescale_factor

###
batch=100
beginning=batch*1000
rescale_factors=[]
summary_statistics=[]
for experiment_id in range(beginning,beginning+1000):
    print(experiment_id)
    params = design.iloc[experiment_id,]
    np.random.seed()
    #simulation of one tree
    tree,vector_count=simulate_bdss_tree_gillespie(params[0],params[2],params[3],params[1],params[4],params[5],params[6],maxTime,1-params[9])
    tree=remove_certain_leaves(tree)
    #SS of the tree
    if vector_count[-1]<100:#if the simulation is a success
        result,rescale_factor=encode_into_summary_statistics(tree,params[5])
    else:
        result,rescale_factor=pd.DataFrame([[0 for k in range(99)]]),0
    rescale_factors.append(rescale_factor)
    summary_statistics.append(result)
    #CBLV of the tree
    #result,rescale_factor=encode_into_most_recent(tree,params[5])
df_summary_statistics=pd.concat(summary_statistics)
df_summary_statistics.index=list(design.index[beginning:beginning+1000])
df_rescale_factors=pd.DataFrame(rescale_factors, index=list(design.index[beginning:beginning+1000]))
name=str(batch)+'.csv'
df_summary_statistics.to_csv(r'C:\Users\LORENA.LAPTOP-LGLJM15L\Documents\Cours ENS\2022-2023\deep_learning\projet\test_set\SS_' +name)
df_rescale_factors.to_csv(r'C:\Users\LORENA.LAPTOP-LGLJM15L\Documents\Cours ENS\2022-2023\deep_learning\projet\test_set\RF_' +name)


###computation of the summary statistics and the rescaling factor for a real tree

with open(r'C:\Users\LORENA.LAPTOP-LGLJM15L\Documents\Cours ENS\2022-2023\deep_learning\projet\real_data\Zurich_HIV_tree.txt', 'r') as file:
    s = file.read() # read from file
    t = Tree(s, format=8) # read Newick format 8
    tree=remove_certain_leaves(t)
    result,rescale_factor=encode_into_summary_statistics(tree,0.25)
    result.to_csv(r'C:\Users\LORENA.LAPTOP-LGLJM15L\Documents\Cours ENS\2022-2023\deep_learning\projet\real_data\Zurich_HIV_SS.csv')

