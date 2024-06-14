import time
import random
from typing import Any

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.tree_search_utils import (
    expand_spectrum,
    list_prunned_actions,
)
from catanatron_experimental.machine_learning.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
    value_production,
)
from catanatron.models.player import Player
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron_gym.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
    resource_hand_features_extended,
    iter_players
)
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)


ALPHABETA_DEFAULT_DEPTH = 1
MAX_SEARCH_TIME_SECS = 20


class MaxnAlphaBetaPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    NOTE: More than 3 levels seems to take much longer, it would be
    interesting to see this with prunning.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false"
        self.value_fn_builder_name = (
            "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.use_value_function = self.value_function
        self.epsilon = epsilon
        self.analyzed_positions = {}
        self.debug = True
    
    def value_function(self, game, p0_color, params=DEFAULT_WEIGHTS):
        """Value function, take 3. Returns a vector of scores. For convenience, scores are ordered by turn"""

        if self.debug:
            print(p0_color)
            self.debug = False

        scores = []

        production_features = build_production_features(True)

        for p, color in iter_players(game.state.colors, p0_color):

            key = f"P{p}"

            our_production_sample = production_features(game, p0_color)
            production = value_production(our_production_sample, key)

            longest_road_length = get_longest_road_length(game.state, color)

            reachability_sample = reachability_features(game, p0_color, 2)
            features = [f"P{p}_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
            reachable_production_at_zero = sum([reachability_sample[f] for f in features])
            features = [f"P{p}_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
            reachable_production_at_one = sum([reachability_sample[f] for f in features])

            # FIX BELOW
            hand_sample = resource_hand_features_extended(game, p0_color)
            features = [f"P{p}_{resource}_IN_HAND" for resource in RESOURCES]
            distance_to_city = (
                max(2 - hand_sample[f"P{p}_WHEAT_IN_HAND"], 0)
                + max(3 - hand_sample[f"P{p}_ORE_IN_HAND"], 0)
            ) / 5.0  # 0 means good. 1 means bad.
            distance_to_settlement = (
                max(1 - hand_sample[f"P{p}_WHEAT_IN_HAND"], 0)
                + max(1 - hand_sample[f"P{p}_SHEEP_IN_HAND"], 0)
                + max(1 - hand_sample[f"P{p}_BRICK_IN_HAND"], 0)
                + max(1 - hand_sample[f"P{p}_WOOD_IN_HAND"], 0)
            ) / 4.0  # 0 means good. 1 means bad.
            hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

            num_in_hand = player_num_resource_cards(game.state, color)
            discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

            # blockability
            buildings = game.state.buildings_by_color[color]
            owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
            owned_tiles = set()
            for n in owned_nodes:
                owned_tiles.update(game.state.board.map.adjacent_tiles[n])
            num_tiles = len(owned_tiles)

            # TODO: Simplify to linear(?)
            num_buildable_nodes = len(game.state.board.buildable_node_ids(color))
            longest_road_factor = (
                params["longest_road"] if num_buildable_nodes == 0 else 0.1
            )

            scores.append(float(
                game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
                + production * params["production"]
                + reachable_production_at_zero * params["reachable_production_0"]
                + reachable_production_at_one * params["reachable_production_1"]
                + hand_synergy * params["hand_synergy"]
                + num_buildable_nodes * params["buildable_nodes"]
                + num_tiles * params["num_tiles"]
                + num_in_hand * params["hand_resources"]
                + discard_penalty
                + longest_road_length * longest_road_factor
                + player_num_dev_cards(game.state, color) * params["hand_devs"]
                + get_played_dev_cards(game.state, color, "KNIGHT") * params["army_size"]
            ))

        # Normalize scores
        total = sum(scores)

        for i in range(len(scores)):
            scores[i] /= total

        diff = game.state.colors.index(p0_color)
        print(f"raws: {scores}")

        # Standardize the score order
        # scores_ordered = []

        # for i in range(len(scores)):
        #     scores_ordered.append(scores[(diff + i) % len(scores)])
        
        return scores
    

    def get_actions(self, game):
        if self.prunning:
            return list_prunned_actions(game)
        return game.state.playable_actions

    def decide(self, game: Game, playable_actions):
        #print(f"Making decision for {game.state.current_color()} for {game.state.current_prompt}")
        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        start = time.time()
        state_id = str(len(game.state.actions))
        node = DebugStateNode(state_id, self.color)  # i think it comes from outside
        deadline = start + MAX_SEARCH_TIME_SECS
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node
        )
        # print("Decision Results:", self.depth, len(actions), time.time() - start)
        # if game.state.num_turns > 10:
        #     render_debug_tree(node)
        #     breakpoint()

        #assert False

        if result[0] is None:
            return playable_actions[0]
        return result[0]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """

        s = "\t"*(2-depth)

        # print(f"{s}alpha: d={depth}")

        # If there is a win, it should have already been discovered
        assert game.winning_color() is None

        # Edge case: If Node is terminal, return static value
        if depth == 0 or time.time() >= deadline:
            values = self.value_function(game, self.color)

            print(f"Seat: {game.state.colors}")
            print(f"Leaf: {values}")

            node.expected_value = values
            # REVIEW should definitely be values
            return None, values

        # Recursive Case: Agent maximizes their own reward function
        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]
        # for action in actions:
        #     print(f"a: {action}")
        #     assert False

        player_idx = game.state.colors.index(game.state.current_player().color)
        num_players = len(game.state.colors)

        best_action = None
        best_value = float("-inf")
        for i, (action, outcomes) in enumerate(action_outcomes.items()):
            action_node = DebugActionNode(action)

            # Find the expected value after taking the action by evaluating all possible resulting boards
            expected_values = [0] * len(game.state.colors)
            for j, (outcome, proba) in enumerate(outcomes):
                out_node = DebugStateNode(
                    f"{node.label} {i} {j}", outcome.state.current_color()
                )

                result = self.alphabeta(
                    outcome, depth - 1, alpha, beta, deadline, out_node
                )
                # print(f"{s}\t\tOut: {result[1]}, {outcome.state.current_color()}")
                # print(f"TEST: {game.state.colors}, {game.state.current_player().color}")
                # print(f"TEST: {game.state.colors.index(game.state.current_player().color)}")
                values = result[1]

                for i in range(num_players):
                    expected_values[i] += proba * values[i]

                action_node.children.append(out_node)
                action_node.probas.append(proba)

            action_node.expected_values = expected_values
            node.children.append(action_node)

            # Optimize for the reward of the player from the turn before
            if expected_values[player_idx] > best_value:
                best_action = action
                best_values = expected_values
            
            # TODO: Add shallow pruning

            # print(f"{s}\tNode: {best_value}")

        node.expected_values = best_values
        return best_action, best_values


class DebugStateNode:
    def __init__(self, label, color):
        self.label = label
        self.children = []  # DebugActionNode[]
        self.expected_value = None
        self.color = color


class DebugActionNode:
    def __init__(self, action):
        self.action = action
        self.expected_value: Any = None
        self.children = []  # DebugStateNode[]
        self.probas = []


def render_debug_tree(node):
    from graphviz import Digraph

    dot = Digraph("AlphaBetaSearch")

    agenda = [node]

    while len(agenda) != 0:
        tmp = agenda.pop()
        dot.node(
            tmp.label,
            label=f"<{tmp.label}<br /><font point-size='10'>{tmp.expected_value}</font>>",
            style="filled",
            fillcolor=tmp.color.value,
        )
        for child in tmp.children:
            action_label = (
                f"{tmp.label} - {str(child.action).replace('<', '').replace('>', '')}"
            )
            dot.node(
                action_label,
                label=f"<{action_label}<br /><font point-size='10'>{child.expected_value}</font>>",
                shape="box",
            )
            dot.edge(tmp.label, action_label)
            for action_child, proba in zip(child.children, child.probas):
                dot.node(
                    action_child.label,
                    label=f"<{action_child.label}<br /><font point-size='10'>{action_child.expected_value}</font>>",
                )
                dot.edge(action_label, action_child.label, label=str(proba))
                agenda.append(action_child)
    print(dot.render())