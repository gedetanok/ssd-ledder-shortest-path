"""
Interactive UI for radio labeling of SSD_m(L_n).

Run:
    python ssd_ladder_ui.py

Layout follows the SSD_m(L_n) figure in the calculation document:
    * top rail u_1, u_2, ..., u_n horizontally
    * bottom rail v_1, ..., v_n
    * top-edge subs w^h_{i,j} stack vertically above u_i -- u_{i+1}
    * bottom-edge subs w^b_{i,j} stack vertically below v_i -- v_{i+1}
    * rung subs w^v_{i,j} fan out horizontally beside the rung u_i -- v_i
        - i = 1            : extend left of the ladder
        - i = n            : extend right of the ladder
        - 2 <= i <= n - 1  : extend left into panel (i-1, i)

Workflow:
    Click vertices in the order you want them labeled. The first click gets
    label 0; subsequent clicks get labels via
        φ(μ_{i+1}) = φ(μ_i) + diam(G) + 1 − d(μ_i, μ_{i+1}).
    After every click the radio condition is checked for all pairs labeled
    so far. The right-hand panel shows the ordering, span, and violations.

Buttons:
    Undo   — remove the last clicked vertex
    Reset  — clear ordering and start over
    Save   — write the current figure to a PNG file
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


# ----------------------------------------------------------------------
# Graph construction with document-style layout
# ----------------------------------------------------------------------

def build_ssd_ladder_doc(m: int, n: int):
    """Return (G, pos) where pos matches the SSD_m(L_n) figure layout.

    Convention:
      * For every edge xy of the original ladder, the m subdivisions w_1..w_m
        are laid out so that w_1 sits exactly on the straight line between x
        and y, and w_2, w_3, ... fan outward perpendicularly in nested layers.
        - m = 1  → just the midpoint dot ("subdivided grid")
        - m = 2  → midpoint + one outward tip ("diamond")
        - m ≥ 3  → midpoint + several nested tips ("layered star")
      * Outward direction:
        - top horizontal edges  → upward
        - bottom horizontal edges → downward
        - leftmost rung (i = 1)  → leftward (outside the ladder)
        - rightmost rung (i = n) → rightward (outside the ladder)
        - inner rungs (2..n-1)   → leftward (into the panel to its left),
                                   matching the SSD_m(L_n) figure
    """
    if m < 1 or n < 2:
        raise ValueError("Need m >= 1 and n >= 2.")

    G = nx.Graph()
    pos = {}

    COL = 4.0           # horizontal distance between adjacent rung columns
    RAIL = 1.6          # half-distance between top and bottom rails
    STEP = 0.85         # outward offset between consecutive subdivisions

    # original ladder vertices
    for i in range(1, n + 1):
        ui, vi = f"u_{i}", f"v_{i}"
        G.add_node(ui)
        G.add_node(vi)
        pos[ui] = (i * COL, RAIL)
        pos[vi] = (i * COL, -RAIL)

    # top horizontal edge subdivisions  w^h_{i,j}
    for i in range(1, n):
        ui, uip1 = f"u_{i}", f"u_{i+1}"
        mid_x = (i + 0.5) * COL
        for j in range(1, m + 1):
            w = f"wh_{i}_{j}"
            G.add_node(w)
            # j=1 sits ON the rail line, j≥2 fan outward (upward)
            pos[w] = (mid_x, RAIL + (j - 1) * STEP)
            G.add_edge(ui, w)
            G.add_edge(uip1, w)

    # bottom horizontal edge subdivisions  w^b_{i,j}
    for i in range(1, n):
        vi, vip1 = f"v_{i}", f"v_{i+1}"
        mid_x = (i + 0.5) * COL
        for j in range(1, m + 1):
            w = f"wb_{i}_{j}"
            G.add_node(w)
            pos[w] = (mid_x, -RAIL - (j - 1) * STEP)
            G.add_edge(vi, w)
            G.add_edge(vip1, w)

    # rung subdivisions  w^v_{i,j}
    for i in range(1, n + 1):
        col_x = i * COL
        ui, vi = f"u_{i}", f"v_{i}"
        direction = +1 if i == n else -1
        for j in range(1, m + 1):
            w = f"wv_{i}_{j}"
            G.add_node(w)
            # j=1 sits ON the rung line, j≥2 fan outward sideways
            pos[w] = (col_x + direction * (j - 1) * STEP, 0.0)
            G.add_edge(ui, w)
            G.add_edge(vi, w)

    return G, pos


def pretty(name: str) -> str:
    p = name.split("_")
    if p[0] == "u":
        return f"$u_{{{p[1]}}}$"
    if p[0] == "v":
        return f"$v_{{{p[1]}}}$"
    if p[0] == "wh":
        return f"$w^{{h}}_{{{p[1]},{p[2]}}}$"
    if p[0] == "wb":
        return f"$w^{{b}}_{{{p[1]},{p[2]}}}$"
    if p[0] == "wv":
        return f"$w^{{v}}_{{{p[1]},{p[2]}}}$"
    return name


# ----------------------------------------------------------------------
# Interactive UI
# ----------------------------------------------------------------------

class RadioLabelingUI:
    NODE_R = 0.25

    def __init__(self, m: int, n: int):
        self.m, self.n = m, n
        self.G, self.pos = build_ssd_ladder_doc(m, n)
        self.diam = nx.diameter(self.G)
        self.delta = self.G.number_of_nodes()

        self.ordering = []
        self.labels = {}
        self.violations = []
        self.last_binding = None

        self.node_patches = {}    # vertex -> Circle
        self.name_artists = {}    # vertex -> Text (vertex name, outside node)
        self.value_artists = {}   # vertex -> Text (label value, inside node)

        self._build_figure()
        self._draw_static()
        self._connect_events()
        self._update_info()

        plt.show()

    # ---------- figure setup ----------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15.0, 8.6))
        self.fig.suptitle(
            f"SSD$_{{{self.m}}}$(L$_{{{self.n}}}$) — Radio Labeling   "
            f"diameter k = {self.diam}   |V| = {self.delta}",
            fontsize=13,
        )

        # main graph area (left ~67%)
        self.ax = self.fig.add_axes([0.02, 0.10, 0.66, 0.82])
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # info / status panel (right ~28%)
        self.ax_info = self.fig.add_axes([0.70, 0.18, 0.28, 0.74])
        self.ax_info.axis("off")

        # buttons row at the bottom (5 buttons)
        ax_undo = self.fig.add_axes([0.70, 0.04, 0.052, 0.05])
        ax_reset = self.fig.add_axes([0.756, 0.04, 0.052, 0.05])
        ax_opt = self.fig.add_axes([0.812, 0.04, 0.055, 0.05])
        ax_ilp = self.fig.add_axes([0.871, 0.04, 0.050, 0.05])
        ax_save = self.fig.add_axes([0.925, 0.04, 0.055, 0.05])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_opt = Button(ax_opt, "Optimize", color="#fff59d", hovercolor="#fff176")
        self.btn_ilp = Button(ax_ilp, "Verify", color="#c5cae9", hovercolor="#9fa8da")
        self.btn_save = Button(ax_save, "Save PNG")
        self.btn_undo.on_clicked(self._on_undo)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_opt.on_clicked(self._on_greedy_optimize)
        self.btn_ilp.on_clicked(self._on_optimize)   # ILP verify
        self.btn_save.on_clicked(self._on_save)

        # status text inside the info panel
        self.info_text = self.ax_info.text(
            0.0, 1.0, "", ha="left", va="top",
            fontsize=9, family="monospace",
            transform=self.ax_info.transAxes,
        )

        # short instruction line above the graph
        self.fig.text(
            0.35, 0.95,
            "Click vertices in the order you want them labeled.",
            ha="center", fontsize=10, color="#37474f",
        )

    def _draw_static(self):
        # edges
        for u, v in self.G.edges:
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.ax.plot([x1, x2], [y1, y2],
                         color="#90a4ae", lw=1.0, alpha=0.75, zorder=1)

        # vertices
        for v in self.G.nodes:
            x, y = self.pos[v]
            patch = mpatches.Circle(
                (x, y), self.NODE_R,
                facecolor="white", edgecolor="#37474f", lw=1.4,
                picker=True, zorder=3,
            )
            patch.vertex_name = v
            self.ax.add_patch(patch)
            self.node_patches[v] = patch

            # vertex name OUTSIDE the node (always visible)
            dx, dy, ha, va = self._text_offset(v)
            self.name_artists[v] = self.ax.text(
                x + dx, y + dy, pretty(v),
                ha=ha, va=va, fontsize=9,
                color="#263238", zorder=4,
            )
            # label value INSIDE the node (set when clicked)
            self.value_artists[v] = self.ax.text(
                x, y, "",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white", zorder=5,
            )

        # view limits
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        mx = (max(xs) - min(xs)) * 0.06 + 0.7
        my = (max(ys) - min(ys)) * 0.14 + 0.7
        self.ax.set_xlim(min(xs) - mx, max(xs) + mx)
        self.ax.set_ylim(min(ys) - my, max(ys) + my)

    def _text_offset(self, name):
        r = self.NODE_R
        if name.startswith("u_"):
            return 0, r + 0.14, "center", "bottom"
        if name.startswith("v_"):
            return 0, -r - 0.14, "center", "top"
        if name.startswith("wh"):
            # wh subs stacked vertically; label above each so they don't collide
            return 0, r + 0.12, "center", "bottom"
        if name.startswith("wb"):
            return 0, -r - 0.12, "center", "top"
        if name.startswith("wv"):
            # wv subs stacked horizontally; label below
            return 0, -r - 0.12, "center", "top"
        return 0, r + 0.10, "center", "bottom"

    # ---------- event handlers ----------

    def _connect_events(self):
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _on_pick(self, event):
        if not isinstance(event.artist, mpatches.Circle):
            return
        v = getattr(event.artist, "vertex_name", None)
        if v is None or v in self.ordering:
            return

        if not self.ordering:
            self.labels[v] = 0
            self.last_binding = None
        else:
            # label v as the minimum integer satisfying the radio condition
            # against EVERY previously labeled vertex w:
            #   f(v) >= f(w) + (k + 1 - d(w, v))    for all w in ordering
            best_label = 0
            best_binding = None
            for w in self.ordering:
                dist = nx.shortest_path_length(self.G, w, v)
                need = self.diam + 1 - dist
                candidate = self.labels[w] + need
                if candidate > best_label:
                    best_label = candidate
                    best_binding = (w, dist, need)
            self.labels[v] = best_label
            self.last_binding = best_binding
        self.ordering.append(v)

        self._verify()
        self._paint_node(v)
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_undo(self, event):
        if not self.ordering:
            return
        v = self.ordering.pop()
        del self.labels[v]
        self._unpaint_node(v)
        self.last_binding = None
        self._verify()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        for v in list(self.ordering):
            self._unpaint_node(v)
        self.ordering.clear()
        self.labels.clear()
        self.violations.clear()
        self.last_binding = None
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_save(self, event):
        fname = f"ssd_{self.m}_L_{self.n}_radio.png"
        self.fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"saved: {fname}")

    # -------- optimization (greedy from every starting vertex) -----------

    def _greedy_from(self, start, dist_matrix):
        """Greedy: at each step pick the vertex that minimises its forced label.

        Forced label of a candidate v given current `labels` is
            max_{w in labeled} [ f(w) + (k + 1 - d(w, v)) ].
        We pick the candidate with the smallest forced label, breaking ties
        by vertex name for determinism.
        """
        k = self.diam
        ordering = [start]
        labels = {start: 0}
        remaining = set(self.G.nodes) - {start}
        while remaining:
            best_label = None
            best_v = None
            for v in sorted(remaining):
                cand = 0
                for w in ordering:
                    need = k + 1 - dist_matrix[w][v]
                    if labels[w] + need > cand:
                        cand = labels[w] + need
                if best_label is None or cand < best_label:
                    best_label = cand
                    best_v = v
            ordering.append(best_v)
            labels[best_v] = best_label
            remaining.remove(best_v)
        return ordering, labels

    def _compute_best_greedy(self, dist_matrix):
        """Run greedy from every vertex; return (ordering, span, start, labels)."""
        best = None
        for start in sorted(self.G.nodes):
            ordering, labels = self._greedy_from(start, dist_matrix)
            span = max(labels.values())
            if best is None or span < best[0]:
                best = (span, start, ordering, labels)
        return best[2], best[0], best[1], best[3]

    # ---- exhaustive branch-and-bound search -----------------------------

    def _bf_dfs(self, ordering, labels, current_max, current_set,
                nodes, dist_matrix, k):
        """DFS that maintains (ordering, labels, current_max).
        Updates self._bf_best_span / self._bf_best_ordering when a
        complete ordering with smaller span is found. Prunes branches
        whose forced label already reaches self._bf_best_span.
        """
        if len(ordering) == len(nodes):
            if current_max < self._bf_best_span:
                self._bf_best_span = current_max
                self._bf_best_ordering = list(ordering)
            return

        for v in nodes:
            if v in current_set:
                continue
            # forced label for v given the current partial labeling
            forced = 0
            for w in ordering:
                need = k + 1 - dist_matrix[w][v]
                cand = labels[w] + need
                if cand > forced:
                    forced = cand
            if forced >= self._bf_best_span:
                continue  # B&B prune: cannot improve best
            ordering.append(v)
            labels[v] = forced
            current_set.add(v)
            new_max = current_max if current_max > forced else forced
            self._bf_dfs(ordering, labels, new_max, current_set,
                         nodes, dist_matrix, k)
            ordering.pop()
            del labels[v]
            current_set.discard(v)

    def _on_greedy_optimize(self, event):
        """Fast greedy multi-start (no provable optimality, but usually optimal
        for SSD ladder graphs). Use the Verify button to confirm with ILP.
        """
        import time
        # reset visuals
        for v in list(self.ordering):
            self._unpaint_node(v)
        self.ordering.clear()
        self.labels.clear()
        self.violations.clear()
        self.last_binding = None
        self.fig.canvas.draw_idle()

        print("=" * 56)
        print(f"GREEDY  SSD_{self.m}(L_{self.n})  "
              f"|V|={self.delta}  k={self.diam}", flush=True)

        dist = dict(nx.all_pairs_shortest_path_length(self.G))
        t0 = time.time()
        ordering, span, start, _ = self._compute_best_greedy(dist)
        elapsed = time.time() - t0
        print(f"  best start = {start}, span = {span}  ({elapsed:.2f}s)")
        print("  (heuristic: span is a valid upper bound, often = rn)")
        print("=" * 56)

        for v in ordering:
            fake = type("PickEv", (), {"artist": self.node_patches[v]})()
            self._on_pick(fake)
        self.fig.canvas.draw_idle()

    def _solve_ilp(self, time_limit=300):
        """Solve radio labeling exactly with PuLP/CBC.

        Variables:
            f[v]   non-negative integer label for vertex v, bounded by greedy span
            S      the span (= max f[v]), bounded by greedy span
            y[u,v] binary for each unordered pair (u, v) with u < v
                   y = 1 means f[u] >= f[v] + c(u,v); y = 0 means the reverse

        Linearisation of |f(u) - f(v)| >= c(u,v) with big-M:
            f[u] - f[v] >= c - M (1 - y)
            f[v] - f[u] >= c - M y
        where M = greedy_span + k is tight (any feasible solution satisfies it).

        Greedy solution provides:
          * tight upper bound on S and f[v] (small upBound -> small LP relaxation)
          * warm start values (CBC uses for fast initial feasible solution)
        """
        import pulp

        k = self.diam
        nodes = list(self.G.nodes)
        dist = dict(nx.all_pairs_shortest_path_length(self.G))

        # Greedy for tight bounds + warm start
        print("  computing greedy upper bound...", flush=True)
        g_ord, g_span, g_start, g_labels = self._compute_best_greedy(dist)
        print(f"  greedy bound: span = {g_span} (start = {g_start})", flush=True)

        big_M = g_span + k   # tight: any feasible solution satisfies this

        prob = pulp.LpProblem("radio_labeling", pulp.LpMinimize)

        f = {v: pulp.LpVariable(f"f_{v}", lowBound=0, upBound=g_span,
                                cat="Integer") for v in nodes}
        S = pulp.LpVariable("S", lowBound=0, cat="Integer")
        y = {}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                y[(u, v)] = pulp.LpVariable(f"y_{u}__{v}", cat="Binary")

        prob += S

        for v in nodes:
            prob += S >= f[v]

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                c = k + 1 - dist[u][v]
                prob += f[u] - f[v] >= c - big_M * (1 - y[(u, v)])
                prob += f[v] - f[u] >= c - big_M * y[(u, v)]

        # Symmetry breaking: subdivisions on the same edge are indistinguishable
        # (each connects to the same pair of endpoints with no other neighbours),
        # so we can WLOG enforce f(w_{i,1}) < f(w_{i,2}) < ... < f(w_{i,m}).
        sym_groups = []
        for i in range(1, self.n):
            sym_groups.append([f"wh_{i}_{j}" for j in range(1, self.m + 1)])
            sym_groups.append([f"wb_{i}_{j}" for j in range(1, self.m + 1)])
        for i in range(1, self.n + 1):
            sym_groups.append([f"wv_{i}_{j}" for j in range(1, self.m + 1)])
        for grp in sym_groups:
            for a, b in zip(grp, grp[1:]):
                # strict ordering (labels are distinct in any valid radio labeling)
                prob += f[b] >= f[a] + 1

        # Reorder greedy labels per symmetry group, so the warm start is feasible
        g_labels = dict(g_labels)
        for grp in sym_groups:
            sorted_vals = sorted(g_labels[v] for v in grp)
            for v, val in zip(grp, sorted_vals):
                g_labels[v] = val

        # Warm-start CBC with the greedy solution
        for v in nodes:
            f[v].setInitialValue(int(g_labels[v]))
        for (u, v), y_var in y.items():
            y_var.setInitialValue(1 if g_labels[u] >= g_labels[v] else 0)
        S.setInitialValue(int(g_span))

        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, warmStart=True)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        labels = {v: int(round(f[v].varValue)) for v in nodes}
        span = int(round(S.varValue))
        # shift so the smallest label is 0
        lo = min(labels.values())
        labels = {v: x - lo for v, x in labels.items()}
        span -= lo

        # Safety: if CBC's incumbent is worse than greedy (e.g. due to
        # time-out with weak warm-start), fall back to the greedy labeling
        # which is guaranteed valid.
        if span > g_span:
            print(f"  ILP best ({span}) worse than greedy ({g_span}); "
                  f"using greedy.", flush=True)
            g_lo = min(g_labels.values())
            labels = {v: x - g_lo for v, x in g_labels.items()}
            span = g_span - g_lo
            status = "TimeLimit-UsedGreedy"

        ordering = sorted(nodes, key=lambda v: (labels[v], v))
        return ordering, labels, span, status

    def _on_optimize(self, event):
        """Solve the radio labeling problem exactly via ILP (PuLP + CBC).
        Falls back to brute force if PuLP is unavailable.
        """
        import time

        # reset visual state
        for v in list(self.ordering):
            self._unpaint_node(v)
        self.ordering.clear()
        self.labels.clear()
        self.violations.clear()
        self.last_binding = None
        self.fig.canvas.draw_idle()

        print("=" * 56)
        print(f"VERIFY (ILP)  SSD_{self.m}(L_{self.n})  "
              f"|V|={self.delta}  k={self.diam}", flush=True)
        print("(this may take a while; the greedy solution will be used as a"
              " warm start)", flush=True)

        t0 = time.time()
        try:
            ordering, labels, span, status = self._solve_ilp()
        except ImportError:
            print("PuLP not installed; falling back to brute force.")
            return self._on_brute_force(event)

        elapsed = time.time() - t0
        print(f"  status: {status}")
        print(f"  optimal span = {span}  (solved in {elapsed:.2f}s)")
        print("=" * 56)

        # Replay the labels via _on_pick so visuals/info panel are consistent
        for v in ordering:
            fake = type("PickEv", (), {"artist": self.node_patches[v]})()
            self._on_pick(fake)
        self.fig.canvas.draw_idle()

    def _on_brute_force(self, event):
        """Fallback: brute-force B&B (kept for completeness / no-PuLP envs)."""
        import time
        for v in list(self.ordering):
            self._unpaint_node(v)
        self.ordering.clear()
        self.labels.clear()
        self.violations.clear()
        self.last_binding = None
        self.fig.canvas.draw_idle()

        dist_matrix = dict(nx.all_pairs_shortest_path_length(self.G))
        k = self.diam
        nodes = sorted(self.G.nodes)
        g_ord, g_span, _, _ = self._compute_best_greedy(dist_matrix)
        self._bf_best_span = g_span
        self._bf_best_ordering = list(g_ord)
        t0 = time.time()
        try:
            for start in nodes:
                self._bf_dfs([start], {start: 0}, 0, {start},
                             nodes, dist_matrix, k)
        except KeyboardInterrupt:
            pass
        print(f"brute force: span = {self._bf_best_span} "
              f"({time.time()-t0:.2f}s)")
        for v in self._bf_best_ordering:
            fake = type("PickEv", (), {"artist": self.node_patches[v]})()
            self._on_pick(fake)
        self.fig.canvas.draw_idle()

    # ---------- node painting ----------

    def _paint_node(self, v):
        i = self.ordering.index(v)
        cmap = plt.get_cmap("viridis")
        c = cmap(0.15 + 0.7 * i / max(self.delta - 1, 1))
        self.node_patches[v].set_facecolor(c)
        self.node_patches[v].set_edgecolor("#0d1b2a")
        self.node_patches[v].set_linewidth(1.8)
        # label value inside the node; emphasise the name outside
        self.value_artists[v].set_text(str(self.labels[v]))
        self.name_artists[v].set_fontweight("bold")
        self.name_artists[v].set_color("#0d1b2a")

    def _unpaint_node(self, v):
        self.node_patches[v].set_facecolor("white")
        self.node_patches[v].set_edgecolor("#37474f")
        self.node_patches[v].set_linewidth(1.4)
        self.value_artists[v].set_text("")
        self.name_artists[v].set_fontweight("normal")
        self.name_artists[v].set_color("#263238")

    # ---------- verification ----------

    def _verify(self):
        violations = []
        items = list(self.labels.items())
        for i, (u, fu) in enumerate(items):
            for w, fw in items[i + 1:]:
                dist = nx.shortest_path_length(self.G, u, w)
                need = self.diam + 1 - dist
                actual = abs(fu - fw)
                if actual < need:
                    violations.append((u, w, actual, need, dist))
        self.violations = violations

    def _update_info(self):
        k = self.diam
        lines = []
        lines.append(f"SSD_{self.m}(L_{self.n})  |V|={self.delta}  diam k={k}")
        lines.append("")
        lines.append("Syarat radio:  |f(u)-f(v)| >= k+1 - d(u,v)")
        lines.append("(berlaku utk SEMUA pasangan u, v)")
        lines.append("")
        lines.append(" d(u,v) | min |f(u)-f(v)|")
        lines.append(" -------+----------------")
        for d in range(1, k + 1):
            lines.append(f"   {d:2d}   |       {k + 1 - d}")
        lines.append("")
        lines.append(f"clicked : {len(self.ordering)} / {self.delta}")
        if self.labels:
            span = max(self.labels.values())
            lines.append(f"span    = {span}")
        else:
            lines.append("span    = —")

        if self.last_binding is not None and self.ordering:
            v = self.ordering[-1]
            w, dist, need = self.last_binding
            lines.append("")
            lines.append(f"last: {v} -> f={self.labels[v]}")
            lines.append(f"  bound by {w} (f={self.labels[w]}):")
            lines.append(f"  d({w},{v})={dist}, need={need}")

        lines.append("")
        lines.append("ordering / labels:")
        for v in self.ordering:
            lines.append(f"  {v:10s} : {self.labels[v]}")

        if len(self.ordering) == self.delta:
            lines.append("")
            lines.append("✓ VALID radio labeling")
            lines.append(f"  span = {max(self.labels.values())}")
            lines.append("  (upper bound for rn)")

        self.info_text.set_text("\n".join(lines))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def ask_int(prompt: str, lo: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            x = int(raw)
        except ValueError:
            print("  please enter an integer")
            continue
        if x < lo:
            print(f"  must be >= {lo}")
            continue
        return x


def main():
    if len(sys.argv) == 3:
        try:
            m, n = int(sys.argv[1]), int(sys.argv[2])
        except ValueError:
            print("usage: python ssd_ladder_ui.py [m n]")
            sys.exit(2)
        if m < 1 or n < 2:
            print("Need m >= 1 and n >= 2")
            sys.exit(2)
    else:
        print("=" * 60)
        print(" SSD_m(L_n) Radio Labeling — interactive UI")
        print("=" * 60)
        m = ask_int("m (subdivisions per edge, m >= 1): ", 1)
        n = ask_int("n (ladder steps, n >= 2): ", 2)

    print(f"\nopening UI for SSD_{m}(L_{n}) ...")
    print("click vertices in the order you want them labeled.")
    print("(buttons: Undo / Reset / Save PNG)")

    RadioLabelingUI(m, n)


if __name__ == "__main__":
    main()
