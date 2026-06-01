"""
SSD explorer — super-subdivision of a ladder graph SSD_k(L_n) OR a prism
graph SSD_k(D_{m,n}) (D_{m,n}=C_m □ P_n: cube/pentagonal-prism/... stacks).
The graph family is chosen interactively at startup.

Two modes:
    1) shortest path   — distance d(s, t) between two vertices (BFS).
    2) radio labeling  — given a user-supplied vertex ordering μ_1, …, μ_δ,
                         assign labels with
                             φ(μ_1) = 0
                             φ(μ_{i+1}) = φ(μ_i) + d + 1 − d(μ_i, μ_{i+1})
                         where d = diam(G); then verify the radio condition
                             |φ(u) − φ(v)| ≥ d + 1 − d(u, v)  for every pair.
                         The span max φ(μ_i) is rn(G) when minimal.

Vertex naming (matches the SSD-ladder document)
-----------------------------------------------
    u_i       top-rail original vertex      (i = 1..n)
    v_i       bottom-rail original vertex   (i = 1..n)
    wh_i_j    j-th subdivision on the top horizontal edge u_i -- u_{i+1}
                                                (i = 1..n-1, j = 1..m)
    wb_i_j    j-th subdivision on the bottom horizontal edge v_i -- v_{i+1}
    wv_i_j    j-th subdivision on the rung u_i -- v_i        (i = 1..n)

Reference:
    Mari, B., & Jeyaraj, R. S. (2023). Radio Labeling of Supersub-Division
    of Path Graphs. IEEE Access, 11.
"""

import re
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------

def build_ssd_ladder(m: int, n: int):
    if m < 1 or n < 2:
        raise ValueError("Need m >= 1 and n >= 2.")

    G = nx.Graph()
    pos, role = {}, {}

    STEP_W, RAIL_H = 5.0, 3.5

    # original ladder vertices
    for i in range(1, n + 1):
        ui, vi = f"u_{i}", f"v_{i}"
        G.add_node(ui)
        G.add_node(vi)
        role[ui] = role[vi] = "orig"
        pos[ui] = (i * STEP_W, RAIL_H)
        pos[vi] = (i * STEP_W, 0.0)

    # top horizontal subdivisions  wh_i_j   (between u_i and u_{i+1})
    for i in range(1, n):
        ui, uip1 = f"u_{i}", f"u_{i+1}"
        mid_x = (pos[ui][0] + pos[uip1][0]) / 2
        for j in range(1, m + 1):
            w = f"wh_{i}_{j}"
            G.add_node(w)
            role[w] = "sub"
            spread = (j - (m + 1) / 2) * 0.55
            lift = 0.8 + 0.55 * (j - 1)
            pos[w] = (mid_x + spread, RAIL_H + lift)
            G.add_edge(ui, w)
            G.add_edge(uip1, w)

    # bottom horizontal subdivisions  wb_i_j   (between v_i and v_{i+1})
    for i in range(1, n):
        vi, vip1 = f"v_{i}", f"v_{i+1}"
        mid_x = (pos[vi][0] + pos[vip1][0]) / 2
        for j in range(1, m + 1):
            w = f"wb_{i}_{j}"
            G.add_node(w)
            role[w] = "sub"
            spread = (j - (m + 1) / 2) * 0.55
            drop = 0.8 + 0.55 * (j - 1)
            pos[w] = (mid_x + spread, 0.0 - drop)
            G.add_edge(vi, w)
            G.add_edge(vip1, w)

    # rung subdivisions  wv_i_j   (between u_i and v_i)
    for i in range(1, n + 1):
        ui, vi = f"u_{i}", f"v_{i}"
        mid_y = (pos[ui][1] + pos[vi][1]) / 2
        for j in range(1, m + 1):
            w = f"wv_{i}_{j}"
            G.add_node(w)
            role[w] = "sub"
            spread = (j - (m + 1) / 2) * 0.7
            pos[w] = (pos[ui][0] + spread, mid_y)
            G.add_edge(ui, w)
            G.add_edge(vi, w)

    return G, pos, role


def build_ssd_prism(k: int, m: int, n: int):
    """Super-subdivision SSD_k(D_{m,n}) of the prism graph D_{m,n}=C_m □ P_n.

    D_{m,n}: n stacked copies of the cycle C_m (m >= 3), corresponding vertices
    of consecutive copies joined by a matching (D_{4,2}=cube, D_{5,2}=pentagonal
    prism, ...). SSD_k replaces every edge by K_{2,k}.

    Vertices:
        p_{c}_{i}      base vertex, cycle position c=1..m, layer i=1..n
        wc_{c}_{i}_{j} j-th sub of cycle edge p_{c}_{i} -- p_{c+1}_{i} (mod m)
        wr_{c}_{i}_{j} j-th sub of vertical edge p_{c}_{i} -- p_{c}_{i+1}
    """
    if k < 1 or m < 3 or n < 2:
        raise ValueError("Need k >= 1, m >= 3 and n >= 2.")

    G = nx.Graph()
    pos, role = {}, {}

    STEP = 0.42

    if m == 3:
        # Flat vertical-stacked triangles (matches the SSD_k(D_{3,n}) figures)
        R = 2.5
        V_GAP = 4.2
        theta0 = math.pi / 2

        def layer_xy(c, i):
            ang = theta0 + 2 * math.pi * (c - 1) / m
            return (R * math.cos(ang), -(i - 1) * V_GAP + R * math.sin(ang))
    else:
        # Oblique 3-D projection (matches cube / pentagonal-prism figures)
        R = 2.4
        SKEW_X = 1.15
        H = 2.7
        theta0 = math.pi / 2 + math.pi / m

        def layer_xy(c, i):
            ang = theta0 + 2 * math.pi * (c - 1) / m
            return (R * math.cos(ang) + (i - 1) * SKEW_X,
                    R * math.sin(ang) + (i - 1) * H)

    for i in range(1, n + 1):
        for c in range(1, m + 1):
            v = f"p_{c}_{i}"
            G.add_node(v)
            role[v] = "orig"
            pos[v] = layer_xy(c, i)

    def add_subs(u, w, tag, c, i):
        ux, uy = pos[u]
        wx, wy = pos[w]
        mx, my = (ux + wx) / 2.0, (uy + wy) / 2.0
        dx, dy = wx - ux, wy - uy
        length = math.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length
        for j in range(1, k + 1):
            off = (j - (k + 1) / 2.0) * STEP
            s = f"{tag}_{c}_{i}_{j}"
            G.add_node(s)
            role[s] = "sub"
            pos[s] = (mx + px * off, my + py * off)
            G.add_edge(u, s)
            G.add_edge(w, s)

    for i in range(1, n + 1):
        for c in range(1, m + 1):
            add_subs(f"p_{c}_{i}", f"p_{c % m + 1}_{i}", "wc", c, i)
    for i in range(1, n):
        for c in range(1, m + 1):
            add_subs(f"p_{c}_{i}", f"p_{c}_{i+1}", "wr", c, i)

    return G, pos, role


# ----------------------------------------------------------------------
# Vertex naming helpers
# ----------------------------------------------------------------------

def pretty(name: str) -> str:
    """Render canonical name as matplotlib mathtext."""
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
    if p[0] == "p":
        return f"$v_{{{p[1]},{p[2]}}}$"
    if p[0] == "wc":
        return f"$w^{{c}}_{{{p[1]},{p[2]},{p[3]}}}$"
    if p[0] == "wr":
        return f"$w^{{r}}_{{{p[1]},{p[2]},{p[3]}}}$"
    return name


def parse_vertex(text: str) -> str:
    """Normalize input forms to canonical underscore form.
    Accepts:  u_1   u1   u 1
              wh_1_2   wh 1 2   w^h_{1,2}   w^h_1,2
              wv_2_1   wv2_1   w^v_{2,1}
    """
    s = text.strip().lower()
    # collapse "w^h" / "w h" patterns into single token "wh","wb","wv","wc","wr"
    s = re.sub(r"w\s*\^?\s*([hbvcr])", r"w\1", s)
    parts = re.findall(r"[a-z]+|\d+", s)
    if not parts:
        raise ValueError(f"could not parse vertex name from {text!r}")
    return "_".join(parts)


# ----------------------------------------------------------------------
# Shortest-path visualization (mode 1)
# ----------------------------------------------------------------------

def _figsize_from_pos(pos, base=11.0, scale=0.42, cap=22.0):
    """Figure size derived from the layout's bounding box (family-agnostic)."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    w = (max(xs) - min(xs)) or 1.0
    h = (max(ys) - min(ys)) or 1.0
    fw = min(cap, max(base, w * scale + 2.0))
    fh = min(cap, max(7.0, h * scale + 2.0))
    return fw, fh


def visualize_path(G, pos, role, source, target, gid):
    path = nx.shortest_path(G, source, target)
    dist = len(path) - 1
    path_edges = list(zip(path, path[1:]))

    fig, ax = plt.subplots(figsize=_figsize_from_pos(pos))

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#cfd8dc", width=0.9, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                           edge_color="#d32f2f", width=2.8)

    orig = [v for v in G.nodes if role[v] == "orig"]
    sub = [v for v in G.nodes if role[v] == "sub"]
    nx.draw_networkx_nodes(G, pos, nodelist=sub, ax=ax,
                           node_color="#e3f2fd", node_size=160,
                           edgecolors="#1976d2", linewidths=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=orig, ax=ax,
                           node_color="#fff8e1", node_size=520,
                           edgecolors="#f57c00", linewidths=1.6)

    p_sub = [v for v in path if role[v] == "sub"]
    p_orig = [v for v in path if role[v] == "orig"]
    nx.draw_networkx_nodes(G, pos, nodelist=p_sub, ax=ax,
                           node_color="#ffcdd2", node_size=210,
                           edgecolors="#d32f2f", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=p_orig, ax=ax,
                           node_color="#ffcdd2", node_size=580,
                           edgecolors="#d32f2f", linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=[source], ax=ax,
                           node_color="#66bb6a", node_size=740,
                           edgecolors="#1b5e20", linewidths=2.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[target], ax=ax,
                           node_color="#ef5350", node_size=740,
                           edgecolors="#b71c1c", linewidths=2.5)

    labels = {v: pretty(v) for v in orig}
    for v in path:
        labels[v] = pretty(v)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9)

    title = (f"{gid}   "
             f"shortest path  {pretty(source)} $\\to$ {pretty(target)}   "
             f"d = {dist}")
    ax.set_title(title, fontsize=13, pad=12)

    path_str = "  $\\to$  ".join(pretty(v) for v in path)
    ax.text(0.5, -0.04, f"path: {path_str}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#555")

    legend = [
        Line2D([0], [0], marker="o", color="w", label="source",
               markerfacecolor="#66bb6a", markeredgecolor="#1b5e20", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="target",
               markerfacecolor="#ef5350", markeredgecolor="#b71c1c", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="original",
               markerfacecolor="#fff8e1", markeredgecolor="#f57c00", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="subdivision",
               markerfacecolor="#e3f2fd", markeredgecolor="#1976d2", markersize=7),
        Line2D([0], [0], color="#d32f2f", lw=2.5, label="shortest path"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return path, dist


# ----------------------------------------------------------------------
# Radio labeling (mode 2)
# ----------------------------------------------------------------------

def radio_labeling(G, ordering, d):
    """Apply  φ(μ_1) = 0,  φ(μ_{i+1}) = φ(μ_i) + d + 1 − dist(μ_i, μ_{i+1})."""
    labels = {ordering[0]: 0}
    for prev, curr in zip(ordering, ordering[1:]):
        dist = nx.shortest_path_length(G, prev, curr)
        labels[curr] = labels[prev] + d + 1 - dist
    return labels


def verify_radio(G, labels, d):
    """Return list of (u, v, |φ(u)-φ(v)|, d + 1 - dist(u,v), dist) violations."""
    violations = []
    items = list(labels.items())
    for i, (u, fu) in enumerate(items):
        for v, fv in items[i + 1:]:
            dist = nx.shortest_path_length(G, u, v)
            need = d + 1 - dist
            actual = abs(fu - fv)
            if actual < need:
                violations.append((u, v, actual, need, dist))
    return violations


def _label_offset(name):
    """Offset (dx, dy in points, ha, va) for the text annotation of each vertex."""
    if name.startswith("u_"):
        return 0, 20, "center", "bottom"
    if name.startswith("v_"):
        return 0, -20, "center", "top"
    if name.startswith("wh"):
        return 14, 0, "left", "center"
    if name.startswith("wb"):
        return 14, 0, "left", "center"
    if name.startswith("wv"):
        return 12, 0, "left", "center"
    # prism / D_{m,n} vertices
    if name.startswith("p_"):
        return 0, 16, "center", "bottom"
    if name.startswith("wc") or name.startswith("wr"):
        return 11, 0, "left", "center"
    return 0, 14, "center", "bottom"


def visualize_labeling(G, pos, role, labels, ordering, d, gid, violations):
    span = max(labels.values())

    fig, ax = plt.subplots(figsize=_figsize_from_pos(pos, base=13.0))

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#b0bec5", width=1.0, alpha=0.8)

    # color nodes by their position in the ordering (light -> dark)
    delta = len(ordering)
    cmap = plt.get_cmap("viridis")
    order_index = {v: i for i, v in enumerate(ordering)}

    orig = [v for v in G.nodes if role[v] == "orig"]
    sub = [v for v in G.nodes if role[v] == "sub"]
    sub_colors = [cmap(0.15 + 0.7 * order_index[v] / max(delta - 1, 1)) for v in sub]
    orig_colors = [cmap(0.15 + 0.7 * order_index[v] / max(delta - 1, 1)) for v in orig]

    nx.draw_networkx_nodes(G, pos, nodelist=sub, ax=ax,
                           node_color=sub_colors, node_size=180,
                           edgecolors="#263238", linewidths=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=orig, ax=ax,
                           node_color=orig_colors, node_size=380,
                           edgecolors="#263238", linewidths=1.2)

    # highlight first (label 0) and last (span) vertices
    first_v = ordering[0]
    last_v = ordering[-1]
    nx.draw_networkx_nodes(G, pos, nodelist=[first_v], ax=ax,
                           node_color="none", node_size=520,
                           edgecolors="#2e7d32", linewidths=2.6)
    nx.draw_networkx_nodes(G, pos, nodelist=[last_v], ax=ax,
                           node_color="none", node_size=520,
                           edgecolors="#c62828", linewidths=2.6)

    # text labels: "<name> : <value>"  beside each vertex
    for v, (x, y) in pos.items():
        dx, dy, ha, va = _label_offset(v)
        ax.annotate(
            f"{pretty(v)} : {labels[v]}",
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=8.5, color="#0d1b2a",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        )

    status = "valid radio labeling ✓" if not violations \
        else f"{len(violations)} radio violation(s) ✗"
    title = (f"{gid} radio labeling   "
             f"d = {d}   span = {span}   {status}")
    ax.set_title(title, fontsize=13, pad=12)

    legend = [
        Line2D([0], [0], marker="o", color="w", label=f"start ({pretty(first_v)} = 0)",
               markerfacecolor=cmap(0.15), markeredgecolor="#2e7d32", markersize=12),
        Line2D([0], [0], marker="o", color="w", label=f"end   ({pretty(last_v)} = {span})",
               markerfacecolor=cmap(0.85), markeredgecolor="#c62828", markersize=12),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Interactive prompts
# ----------------------------------------------------------------------

HELP_NAMING = """
Vertex name format
------------------
  u_i       top-rail original vertex,    i in {1..n}
  v_i       bottom-rail original vertex
  wh_i_j    j-th sub on top edge u_i -- u_{i+1},     i in {1..n-1}, j in {1..m}
  wb_i_j    j-th sub on bottom edge v_i -- v_{i+1}
  wv_i_j    j-th sub on rung u_i -- v_i,              i in {1..n}

Accepted input forms:  u_1   u1   wh_1_2   w^h_{1,2}   wv 2 1
"""


HELP_NAMING_PRISM = """
Vertex name format (D_{m,n} = prism C_m x P_n)
----------------------------------------------
  p_c_i      base vertex: cycle position c in {1..m}, layer i in {1..n}
  wc_c_i_j   j-th sub on cycle edge p_c_i -- p_(c+1)_i,  j in {1..k}
  wr_c_i_j   j-th sub on vertical edge p_c_i -- p_c_(i+1), i in {1..n-1}

Accepted input forms:  p_1_2   p 1 2   wc_1_1_1   w^c_{1,1,1}   wr 2 1 1
"""


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


def list_vertices(m, n):
    """Print ladder vertex names grouped by type, in a stable order."""
    orig_top = [f"u_{i}" for i in range(1, n + 1)]
    orig_bot = [f"v_{i}" for i in range(1, n + 1)]
    wh = [f"wh_{i}_{j}" for i in range(1, n) for j in range(1, m + 1)]
    wb = [f"wb_{i}_{j}" for i in range(1, n) for j in range(1, m + 1)]
    wv = [f"wv_{i}_{j}" for i in range(1, n + 1) for j in range(1, m + 1)]
    print("Vertices:")
    print("  top rail   :", "  ".join(orig_top))
    print("  bottom rail:", "  ".join(orig_bot))
    if wh:
        print("  top subs   :", "  ".join(wh))
    if wb:
        print("  bot subs   :", "  ".join(wb))
    print("  rung subs  :", "  ".join(wv))


def list_vertices_prism(k, m, n):
    """Print prism D_{m,n} vertex names grouped by type."""
    base = [f"p_{c}_{i}" for i in range(1, n + 1) for c in range(1, m + 1)]
    wc = [f"wc_{c}_{i}_{j}" for i in range(1, n + 1)
          for c in range(1, m + 1) for j in range(1, k + 1)]
    wr = [f"wr_{c}_{i}_{j}" for i in range(1, n)
          for c in range(1, m + 1) for j in range(1, k + 1)]
    print("Vertices:")
    print("  base       :", "  ".join(base))
    print("  cycle subs :", "  ".join(wc))
    if wr:
        print("  rung subs  :", "  ".join(wr))


def ask_vertex(G, prompt):
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  empty input; try again")
            continue
        try:
            name = parse_vertex(raw)
        except ValueError as e:
            print(f"  {e}; try again")
            continue
        if name not in G:
            print(f"  '{name}' is not a vertex of the graph; try again")
            continue
        return name


def ask_ordering(spec):
    G = spec["G"]
    list_fn = spec["list_fn"]
    delta = G.number_of_nodes()
    print(f"\nEnter the ordering μ_1, μ_2, …, μ_{delta} of all {delta} vertices.")
    print("Separate names with commas or whitespace. Type 'list' to see all vertex")
    print("names, or 'auto' for a quick greedy heuristic to seed an ordering.\n")

    while True:
        raw = input("ordering: ").strip()
        if raw.lower() == "list":
            list_fn()
            continue
        if raw.lower() == "auto":
            order = greedy_ordering(G)
            print("  greedy ordering:", ", ".join(order))
            confirm = input("  use this ordering? [Y/n] ").strip().lower()
            if confirm in ("", "y", "yes"):
                return order
            continue

        tokens = [t for t in raw.replace(",", " ").split() if t]
        try:
            ordering = [parse_vertex(t) for t in tokens]
        except ValueError as e:
            print(f"  parse error: {e}; try again")
            continue

        if len(ordering) != delta:
            print(f"  expected {delta} vertices, got {len(ordering)}")
            continue
        if len(set(ordering)) != len(ordering):
            seen, dups = set(), set()
            for v in ordering:
                if v in seen:
                    dups.add(v)
                seen.add(v)
            print(f"  duplicate vertices: {sorted(dups)}")
            continue
        unknown = [v for v in ordering if v not in G]
        if unknown:
            print(f"  unknown vertices: {unknown}")
            continue
        missing = set(G.nodes) - set(ordering)
        if missing:
            print(f"  missing vertices: {sorted(missing)}")
            continue
        return ordering


def greedy_ordering(G):
    """Heuristic seed: at each step pick a not-yet-placed vertex farthest from the previous."""
    start = next(iter(G.nodes))  # any vertex; tweak if you want to start elsewhere
    if "u_1" in G:
        start = "u_1"
    elif "p_1_1" in G:
        start = "p_1_1"
    placed = [start]
    remaining = set(G.nodes) - {start}
    while remaining:
        prev = placed[-1]
        # farthest from prev (break ties by name for determinism)
        best = max(remaining,
                   key=lambda v: (nx.shortest_path_length(G, prev, v), v))
        placed.append(best)
        remaining.remove(best)
    return placed


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _base_diameter(G, role) -> int:
    """Diameter over ORIGINAL vertices only (thesis convention): the largest
    shortest-path distance between two non-subdivision vertices (= 2 x diam of
    the base graph). For prisms with odd m this is one less than nx.diameter(G)
    because there a subdivision vertex is the eccentric one; the document uses
    this original-vertex value.
    """
    base = [v for v in G if role[v] == "orig"]
    best = 0
    for u in base:
        dist = nx.single_source_shortest_path_length(G, u)
        for v in base:
            if dist[v] > best:
                best = dist[v]
    return best


def make_ladder_spec_cli(k, n):
    G, pos, role = build_ssd_ladder(k, n)
    return {
        "G": G, "pos": pos, "role": role,
        "gid": f"SSD$_{{{k}}}$(L$_{{{n}}}$)",
        "name_id": f"SSD_{k}(L_{n})",
        "help": HELP_NAMING,
        "list_fn": lambda: list_vertices(k, n),
        "diam": _base_diameter(G, role),
    }


def make_prism_spec_cli(k, m, n):
    G, pos, role = build_ssd_prism(k, m, n)
    return {
        "G": G, "pos": pos, "role": role,
        "gid": f"SSD$_{{{k}}}$(D$_{{{m},{n}}}$)",
        "name_id": f"SSD_{k}(D_{m},{n})",
        "help": HELP_NAMING_PRISM,
        "list_fn": lambda: list_vertices_prism(k, m, n),
        "diam": _base_diameter(G, role),
    }


def run_shortest_path(spec):
    G, pos, role = spec["G"], spec["pos"], spec["role"]
    print(spec["help"])
    spec["list_fn"]()
    print()
    src = ask_vertex(G, "source vertex: ")
    tgt = ask_vertex(G, "target vertex: ")
    path, dist = visualize_path(G, pos, role, src, tgt, spec["gid"])
    print(f"\nshortest distance d({src}, {tgt}) = {dist}")
    print("path:", " -> ".join(path))


def run_radio_labeling(spec):
    G, pos, role = spec["G"], spec["pos"], spec["role"]
    d = spec["diam"]
    print(f"\ndiameter d = {d}  (jarak terjauh antar titik asli)")
    print(spec["help"])
    spec["list_fn"]()

    ordering = ask_ordering(spec)
    labels = radio_labeling(G, ordering, d)
    span = max(labels.values())
    violations = verify_radio(G, labels, d)

    print("\nlabels (in user-supplied order):")
    for v in ordering:
        print(f"  φ({v}) = {labels[v]}")
    print(f"\nspan = max φ = {span}")
    print(f"radio violations: {len(violations)}")
    if violations:
        print("first 10:")
        for u, v, actual, need, dist in violations[:10]:
            print(f"  ({u}, {v}): |dφ| = {actual} < required {need}  (dist = {dist})")
    else:
        print(f"→ valid radio labeling: span is an upper bound "
              f"for rn({spec['name_id']})")

    visualize_labeling(G, pos, role, labels, ordering, d, spec["gid"], violations)


def main():
    print("=" * 66)
    print(" SSD explorer — shortest path & radio labeling")
    print("=" * 66)
    print("\nPilih bentuk graf:")
    print("  1) SSD_k(L_n)    — super sub-divisi graf Tangga (Ladder)")
    print("  2) SSD_k(D_m,n)  — super sub-divisi graf Prisma/Tabung (C_m x P_n)")
    gchoice = (input("pilih graf [1/2] (default 1): ").strip() or "1")

    print("\nMode:")
    print("  1) shortest path between two vertices")
    print("  2) radio labeling with user-supplied vertex ordering")
    mode = (input("choose [1/2] (default 2): ").strip() or "2")

    if gchoice == "2":
        k = ask_int("k (subdivisi per rusuk, k >= 1): ", 1)
        m = ask_int("m (ukuran siklus C_m, m >= 3): ", 3)
        n = ask_int("n (jumlah lapis, n >= 2): ", 2)
        spec = make_prism_spec_cli(k, m, n)
    else:
        k = ask_int("k (subdivisi per rusuk, k >= 1): ", 1)
        n = ask_int("n (langkah ladder, n >= 2): ", 2)
        spec = make_ladder_spec_cli(k, n)

    G = spec["G"]
    print(f"\n{spec['name_id']}:  |V| = {G.number_of_nodes()},  "
          f"|E| = {G.number_of_edges()}")

    if mode == "1":
        run_shortest_path(spec)
    else:
        run_radio_labeling(spec)


if __name__ == "__main__":
    main()
