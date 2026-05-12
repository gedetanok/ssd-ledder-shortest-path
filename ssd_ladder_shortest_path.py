"""
SSD_m(L_n) explorer — super-subdivision of a ladder graph.

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
    return name


def parse_vertex(text: str) -> str:
    """Normalize input forms to canonical underscore form.
    Accepts:  u_1   u1   u 1
              wh_1_2   wh 1 2   w^h_{1,2}   w^h_1,2
              wv_2_1   wv2_1   w^v_{2,1}
    """
    s = text.strip().lower()
    # collapse "w^h" / "w h" patterns into single token "wh", "wb", "wv"
    s = re.sub(r"w\s*\^?\s*([hbv])", r"w\1", s)
    parts = re.findall(r"[a-z]+|\d+", s)
    if not parts:
        raise ValueError(f"could not parse vertex name from {text!r}")
    return "_".join(parts)


# ----------------------------------------------------------------------
# Shortest-path visualization (mode 1)
# ----------------------------------------------------------------------

def visualize_path(G, pos, role, source, target, m, n):
    path = nx.shortest_path(G, source, target)
    dist = len(path) - 1
    path_edges = list(zip(path, path[1:]))

    fig_w = max(11.0, 2.6 * n + 1.0 * m)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))

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

    title = (f"SSD$_{{{m}}}$(L$_{{{n}}}$)   "
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
    return 0, 14, "center", "bottom"


def visualize_labeling(G, pos, role, labels, ordering, d, m, n, violations):
    span = max(labels.values())

    fig_w = max(13.0, 3.0 * n + 1.4 * m)
    fig_h = max(7.5, 3.5 + 1.0 * m)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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
    title = (f"SSD$_{{{m}}}$(L$_{{{n}}}$) radio labeling   "
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
    """Print vertex names grouped by type, in a stable order."""
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


def ask_ordering(G, m, n):
    delta = G.number_of_nodes()
    print(f"\nEnter the ordering μ_1, μ_2, …, μ_{delta} of all {delta} vertices.")
    print("Separate names with commas or whitespace. Type 'list' to see all vertex")
    print("names, or 'auto' for a quick greedy heuristic to seed an ordering.\n")

    while True:
        raw = input("ordering: ").strip()
        if raw.lower() == "list":
            list_vertices(m, n)
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
    start = next(iter(G.nodes))  # any vertex; tweak if you want to start from u_1
    if "u_1" in G:
        start = "u_1"
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

def run_shortest_path(G, pos, role, m, n):
    print(HELP_NAMING)
    list_vertices(m, n)
    print()
    src = ask_vertex(G, "source vertex: ")
    tgt = ask_vertex(G, "target vertex: ")
    path, dist = visualize_path(G, pos, role, src, tgt, m, n)
    print(f"\nshortest distance d({src}, {tgt}) = {dist}")
    print("path:", " -> ".join(path))


def run_radio_labeling(G, pos, role, m, n):
    d = nx.diameter(G)
    print(f"\ndiameter d = {d}")
    print(HELP_NAMING)
    list_vertices(m, n)

    ordering = ask_ordering(G, m, n)
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
        print("→ valid radio labeling: span is an upper bound for rn(SSD_m(L_n))")

    visualize_labeling(G, pos, role, labels, ordering, d, m, n, violations)


def main():
    print("=" * 66)
    print(" SSD_m(L_n) explorer — shortest path & radio labeling")
    print("=" * 66)

    print("\nMode:")
    print("  1) shortest path between two vertices")
    print("  2) radio labeling with user-supplied vertex ordering")
    mode = (input("choose [1/2] (default 2): ").strip() or "2")

    m = ask_int("m (subdivisions per edge, m >= 1): ", 1)
    n = ask_int("n (ladder steps, n >= 2): ", 2)

    G, pos, role = build_ssd_ladder(m, n)
    print(f"\n|V| = {G.number_of_nodes()},  |E| = {G.number_of_edges()}")

    if mode == "1":
        run_shortest_path(G, pos, role, m, n)
    else:
        run_radio_labeling(G, pos, role, m, n)


if __name__ == "__main__":
    main()
