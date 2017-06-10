#!/usr/bin/env python

"""
    Authors: Zach Byrceland and Dario Marasco
    Date: June 2017

    This program is used to compare Co-op Matching to Deferred Acceptance Matching
"""

import copy
import itertools
from random import shuffle
from timeit import default_timer as timer

import pandas as pd
import numpy as np

class Entity(object):
    """ represents an agent """
    def __init__(self, id):
        self.id = id
        # rankings indexed by agent
        self.rankings = {}
        # agents indexed by rankings
        self.preferences = {}
        
    def generate_rankings(self, n):
        tmp = range(0, n)
        shuffle(tmp)
        for i in range(0, n):
            self.rankings[i] = tmp[i]
        self.preferences = {v: k for k, v in self.rankings.iteritems()}

    def __getitem__(self, i):
        return self.rankings[i]


def deferred_acceptance(lhs, rhs, agent_id=None, agent_side=None):
    """ Finds a stable matching between two disjoint sets of agents 
        lhs (students) and rhs(employers)
        sets do not need to equally sized
        """

    l_rhs = len(rhs)
    l_lhs = len(lhs)
    pairs = [[] for _ in xrange(l_rhs)]
    enqueued = range(0, len(lhs))
    best_option = [0] * len(lhs)
    new_match = None

    while len(enqueued) != 0:
        for j in range(0, l_lhs):
            if j in enqueued:
                pairs[lhs[j].preferences[best_option[j]]].append(lhs[j].id)
                enqueued.remove(j)
        for j in range(0, l_rhs):
            if len(pairs[j]) > 1:
                for i in range(len(rhs[j].preferences) - 1, -1, -1):
                    if rhs[j].preferences[i] in pairs[j]:
                        pairs[j].remove(rhs[j].preferences[i])
                        best_option[rhs[j].preferences[i]] += 1
                        # difficult edge case
                        if best_option[rhs[j].preferences[i]] < l_rhs:
                            enqueued.append(rhs[j].preferences[i])
                    if len(pairs[j]) == 1:
                        break

    d_pairs = []
    for e, s in enumerate(pairs):
        student_cost = 0
        employer_cost = 0
        for j in range(0, len(rhs)):
            if len(s) > 0 and lhs[s[0]].preferences[j] == e:
                student_cost = j
                if agent_side == lhs and lhs[s[0]].id == agent_id:
                    new_match = rhs[e].id
                break
        for j in range(0, len(lhs)):
            if len(s) > 0 and rhs[e].preferences[j] == s[0]:
                employer_cost = j
                if agent_side == rhs and rhs[e].id == agent_id:
                    new_match = lhs[s[0]].id
                break
        if len(s) > 0:
            d_pairs.append((s[0], e, student_cost + employer_cost))

    total_cost = sum([_[2] for _ in d_pairs])

    if agent_id is not None:
        return total_cost, d_pairs, new_match
    else:
        return total_cost, d_pairs

def coop_matching(lhs, rhs, agent_id=None, agent_side=None):
    """ Mimics the Drexel co-op pairing system
        Iterates over ever possible sum of rankings and
        then pairs the minimum value. Not stable, not truthful
        "ties" are given to the lower ranking lhs and rhs agents
        """

    l_rhs = len(rhs)
    l_lhs = len(lhs)
    max_sum = l_lhs + l_rhs
    pairs = []
    a_s = range(l_lhs) # available students
    a_e = range(l_rhs) # " " employers
    new_match = None

    costs_lookup = {}
    for i in range(l_lhs):
        for j in range(l_rhs):
            costs_lookup[(i, j)] = lhs[i][j] + rhs[j][i]

    for current_cost in range(0, max_sum): #iterate over all possible sums
        for i in range(l_lhs):
            for j in range(l_rhs):
                cost = costs_lookup[(i, j)] # cost of a given pair
                # if both agents are available and
                # if cost is under the current iteration value
                if cost <= current_cost and i in a_s and j in a_e:
                    if agent_side == lhs and lhs[i].id == agent_id:
                        new_match = rhs[j].id
                    elif agent_side == rhs and rhs[j].id == agent_id:
                        new_match = lhs[i].id
                    # create match
                    pairs.append((i, j, cost))
                    a_s.remove(i)
                    a_e.remove(j)
        if len(a_s) == 0 or len(a_e) == 0: # stop if no agents are left
            break

    total_cost = sum([_[2] for _ in pairs])

    if agent_id is not None:
        return total_cost, pairs, new_match
    else:
        return total_cost, pairs

def incentive_to_lie(algorithm, lhs, rhs, agent_id, agent_side):
    pref_perm_list = None
    other_side = -1
    if agent_side == lhs:
        other_side = len(rhs)
    else:
        other_side = len(lhs)
    pref_perm_list = itertools.permutations(range(0, other_side))

    # find the agent under analysis in lhs
    agent = agent_side[agent_id]

    # calculate original costs
    [alg_cost, alg_pairs, alg_agent] = algorithm(lhs, rhs, agent_id, agent_side)
    if alg_agent is None:
        return 0, None

    alg_agent_cost = None
    if algorithm == deferred_acceptance:
        for da_pref in agent.preferences:
            if agent.preferences[da_pref] == alg_agent:
                alg_agent_cost = da_pref
                break
    elif algorithm == coop_matching:
        alg_agent_cost = agent.rankings[alg_agent]

    # save original preferences / rankings
    orig_pref = None
    if algorithm == deferred_acceptance:
        orig_pref = copy.deepcopy(agent.preferences)
    elif algorithm == coop_matching:
        orig_pref = copy.deepcopy(agent.rankings)

    # remove original preferences from set of permutations / rankings
    #pref_perm.remove(orig_pref)

    # iterate over remaining permutations, checking for cost reductions
    min_cost = copy.copy(alg_agent_cost)
    best_lie = None
    #lie_agent = None

    for p in pref_perm_list:
        pref = {x[0]:x[1] for x in zip(range(other_side), p)}
        if algorithm == deferred_acceptance:
            agent.preferences = pref
        elif algorithm == coop_matching:
            agent.rankings = pref
        [perm_cost, perm_pairs, perm_agent] = algorithm(lhs, rhs, agent_id, agent_side)
        if perm_agent is None:
            continue

        perm_agent_cost = None
        if algorithm == deferred_acceptance:
            for da_pref in orig_pref:
                if orig_pref[da_pref] == perm_agent:
                    perm_agent_cost = da_pref
                    break
        elif algorithm == coop_matching:
            perm_agent_cost = orig_pref[perm_agent]


        if min_cost > perm_agent_cost:
            min_cost = copy.copy(perm_agent_cost)
            best_lie = pref
            #lie_agent = perm_agent

    # reset preferences / rankings
    if algorithm == deferred_acceptance:
        agent.preferences = orig_pref
    elif algorithm == coop_matching:
        agent.rankings = orig_pref

    # calculate incentive to lie
    incentive = 0
    if alg_agent_cost is not None or min_cost is not None:
        incentive = alg_agent_cost - min_cost

    # return incentive and best lie
    return incentive, best_lie

def generate_entities(n, num_ranking):
    """ Generates an Entity object with n rankings and preferences """
    ret = {}
    for i in range(n):
        ret[i] = Entity(i)
        ret[i].generate_rankings(num_ranking)
    return ret

def main():

    num_tests = 100

    #test size
    #sizes = [(4,4)]

    # speed tests, don't run these when checking for incentive to lie
    sizes = [(4,4), (8,8), (16,16), (32,32), (64, 64), (128, 128), (256, 256), (512,512), (1024,1024), (2048,2048)]


    # full tests, these are expensive
    #sizes = [(8,8), (9,9), (10, 10)]

    # fast tests
    #sizes = [(4,4), (7,3), (8,2), (8,4)] 
    #sizes = [(4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (6,2), (6,4), (6,6), (8,2), (8,4), (8,6), (2, 8), (6,8)] 

    test_lie = False
    test_blocking = True
    show_grid = False

    for size in sizes:
        print "SIZE (s,e): ", size
        num_students = size[0]
        num_employers = size[1]
        data = {
            "coop_cost": [],
            "coop_time": [],
            "da_time": [],
            "da_cost": [],
        }

        if test_lie:
            data["avg_s_lie_coop"] = []
            data["avg_e_lie_coop"] = []
            data["avg_e_lie_da"] = []
        if test_blocking:
            data["num_coop_blocking_pairs"] = []
            data["avg_coop_blocking_pairs_imp"] = []

            

        for t in range(num_tests):
            s = generate_entities(num_students, num_employers)
            e = generate_entities(num_employers, num_students)

            if show_grid:
                print "Grid"
                tmp = "\t"
                for j in range(num_employers):
                    tmp += "e%s\t" % (j)
                print tmp

                for i in range(num_students):
                    tmp = "s%s\t" % (i)
                    for j in range(num_employers):
                        tmp += "(%s,%s)\t"  % (s[i].rankings[j], e[j].rankings[i])
                    print tmp
                print "--------------------\nResults:"

            start = timer()
            [cost_coop, pairs_coop] = coop_matching(s, e)
            end = timer()
            data['coop_time'].append(end - start)
            data['coop_cost'].append(cost_coop)

            if test_blocking:
                blocking_pairs = []
                blocking_imp = []

                # Check all possible swaps in resulting pairs to see if a better result can be achieved
                for pair in pairs_coop:
                    for opair in pairs_coop:
                        if pair != opair:
                            total_cost = pair[2] + opair[2]
                            s1 = pair[0]
                            s2 = opair[0]
                            e1 = pair[1]
                            e2 = opair[1]
                            new_cost = s[s1][e2] + s[s2][e1] + e[e1][s2] + e[e2][s1]
                            #check if both pairs are strictly better off
                            if e[e2][s1] <= e[e2][s2] and s[s1][e2] <= s[s1][e1] and e[e1][s2] <= e[e1][s1] and s[s2][e1] <= s[s2][e2]:
                            # check if both pairs agents are better off in some way (but not each individual agent)
                            #if new_cost < total_cost:
                                blocking_pairs.append((pair, opair))
                                blocking_imp.append(total_cost - new_cost)
                num_bp = len(blocking_pairs) / 2
                data['num_coop_blocking_pairs'].append(num_bp) # since a pair shows up twice
                if num_bp > 0:
                    data['avg_coop_blocking_pairs_imp'].append(np.mean(blocking_imp)) # since a pair shows up twice
                else:
                    data['avg_coop_blocking_pairs_imp'].append(0) # since a pair shows up twice

            start = timer()
            [cost_def, pairs_def] = deferred_acceptance(s, e)
            end = timer()
            data['da_cost'].append(cost_def)
            data['da_time'].append(end - start)

            if test_lie:
                # test employer lies for co-op and DA
                avg_e_lie_da = []
                avg_e_lie_coop = []
                for y in range(0, num_employers):
                    [i_da, _] = incentive_to_lie(deferred_acceptance, s, e, y, e)
                    [i_coop, _] = incentive_to_lie(coop_matching, s, e, y, e)
                    avg_e_lie_da.append(i_da)
                    avg_e_lie_coop.append(i_coop)

                # test student lies for co-op matching
                avg_s_lie_coop = []
                for y in range(0, num_students):
                    [i, _] = incentive_to_lie(deferred_acceptance, s, e, y, s)
                    avg_s_lie_coop.append(i_coop)

                data["avg_s_lie_coop"].append(np.mean(avg_s_lie_coop))
                data["avg_e_lie_coop"].append(np.mean(avg_e_lie_coop))
                data["avg_e_lie_da"].append(np.mean(avg_e_lie_da))

        # show results
        print pd.DataFrame(data).describe()

if __name__ == '__main__':
    main()

