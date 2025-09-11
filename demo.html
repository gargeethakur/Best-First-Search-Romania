import React, { useMemo, useRef, useState, useEffect } from "react";
import { Play, RotateCcw } from "lucide-react";

// --- Data: Romania Map Graph (classic AI problem) ---
// Coordinates are rough, normalized to a 1000x700 canvas for layout clarity.
const POS = {
  Arad: { x: 120, y: 300 },
  Zerind: { x: 160, y: 220 },
  Oradea: { x: 200, y: 160 },
  Timisoara: { x: 80, y: 420 },
  Lugoj: { x: 160, y: 460 },
  Mehadia: { x: 200, y: 520 },
  Drobeta: { x: 180, y: 600 },
  Sibiu: { x: 300, y: 260 },
  Rimnicu_Vilcea: { x: 380, y: 320 },
  Fagaras: { x: 380, y: 220 },
  Craiova: { x: 380, y: 520 },
  Pitesti: { x: 480, y: 360 },
  Bucharest: { x: 600, y: 420 },
  Giurgiu: { x: 620, y: 520 },
  Urziceni: { x: 720, y: 380 },
  Hirsova: { x: 820, y: 360 },
  Eforie: { x: 880, y: 440 },
  Vaslui: { x: 780, y: 300 },
  Iasi: { x: 760, y: 220 },
  Neamt: { x: 680, y: 200 },
};

// Helper to make names user-friendly
const label = (name) => name.replace(/_/g, " ");

// Undirected weighted edges (subset matches the standard Romania graph)
const EDGES = {
  Arad: [
    { to: "Zerind", w: 75 },
    { to: "Timisoara", w: 118 },
    { to: "Sibiu", w: 140 },
  ],
  Zerind: [
    { to: "Arad", w: 75 },
    { to: "Oradea", w: 71 },
  ],
  Oradea: [
    { to: "Zerind", w: 71 },
    { to: "Sibiu", w: 151 },
  ],
  Timisoara: [
    { to: "Arad", w: 118 },
    { to: "Lugoj", w: 111 },
  ],
  Lugoj: [
    { to: "Timisoara", w: 111 },
    { to: "Mehadia", w: 70 },
  ],
  Mehadia: [
    { to: "Lugoj", w: 70 },
    { to: "Drobeta", w: 75 },
  ],
  Drobeta: [
    { to: "Mehadia", w: 75 },
    { to: "Craiova", w: 120 },
  ],
  Craiova: [
    { to: "Drobeta", w: 120 },
    { to: "Rimnicu_Vilcea", w: 146 },
    { to: "Pitesti", w: 138 },
  ],
  Sibiu: [
    { to: "Arad", w: 140 },
    { to: "Oradea", w: 151 },
    { to: "Fagaras", w: 99 },
    { to: "Rimnicu_Vilcea", w: 80 },
  ],
  Rimnicu_Vilcea: [
    { to: "Sibiu", w: 80 },
    { to: "Craiova", w: 146 },
    { to: "Pitesti", w: 97 },
  ],
  Fagaras: [
    { to: "Sibiu", w: 99 },
    { to: "Bucharest", w: 211 },
  ],
  Pitesti: [
    { to: "Rimnicu_Vilcea", w: 97 },
    { to: "Craiova", w: 138 },
    { to: "Bucharest", w: 101 },
  ],
  Bucharest: [
    { to: "Fagaras", w: 211 },
    { to: "Pitesti", w: 101 },
    { to: "Giurgiu", w: 90 },
    { to: "Urziceni", w: 85 },
  ],
  Giurgiu: [
    { to: "Bucharest", w: 90 },
  ],
  Urziceni: [
    { to: "Bucharest", w: 85 },
    { to: "Hirsova", w: 98 },
    { to: "Vaslui", w: 142 },
  ],
  Hirsova: [
    { to: "Urziceni", w: 98 },
    { to: "Eforie", w: 86 },
  ],
  Eforie: [
    { to: "Hirsova", w: 86 },
  ],
  Vaslui: [
    { to: "Urziceni", w: 142 },
    { to: "Iasi", w: 92 },
  ],
  Iasi: [
    { to: "Vaslui", w: 92 },
    { to: "Neamt", w: 87 },
  ],
  Neamt: [
    { to: "Iasi", w: 87 },
  ],
};

// Utility: Euclidean distance heuristic using positions
const h = (a, b) => {
  const A = POS[a];
  const B = POS[b];
  if (!A || !B) return Infinity;
  const dx = A.x - B.x;
  const dy = A.y - B.y;
  return Math.sqrt(dx * dx + dy * dy);
};

// Greedy Best-First Search (minimize h(n) to goal). Returns {expandedOrder, parents, goal}
const greedyBestFirst = (start, goal) => {
  const frontier = [start];
  const cameFrom = { [start]: null };
  const visited = new Set();
  const expandedOrder = [];

  while (frontier.length > 0) {
    // pick node with smallest heuristic
    frontier.sort((a, b) => h(a, goal) - h(b, goal));
    const current = frontier.shift();
    if (!current) break;
    if (visited.has(current)) continue;

    visited.add(current);
    expandedOrder.push(current);

    if (current === goal) {
      return { expandedOrder, parents: cameFrom, goal };
    }

    for (const { to } of EDGES[current] || []) {
      if (!visited.has(to) && !(to in cameFrom)) {
        cameFrom[to] = current;
        frontier.push(to);
      }
    }
  }
  return { expandedOrder, parents: cameFrom, goal: null };
};

const buildPath = (parents, goal) => {
  if (!goal) return [];
  const path = [];
  let cur = goal;
  while (cur !== null && cur !== undefined) {
    path.push(cur);
    cur = parents[cur] ?? null;
  }
  return path.reverse();
};

// Palette for stepwise expansion (distinct colors). Final path = BLUE.
const STEP_COLORS = [
  "#ef4444", // red-500
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#8b5cf6", // violet-500
  "#14b8a6", // teal-500
  "#eab308", // yellow-500
  "#f97316", // orange-500
  "#22c55e", // green-500
  "#06b6d4", // cyan-500
  "#a855f7", // purple-500
];

export default function RomaniaGBFS() {
  const cities = useMemo(() => Object.keys(POS), []);
  const [start, setStart] = useState("Arad");
  const [goal, setGoal] = useState("Bucharest");
  const [running, setRunning] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState(-1);
  const [expandedOrder, setExpandedOrder] = useState([]);
  const [finalPath, setFinalPath] = useState([]);
  const [parents, setParents] = useState({});
  const [speed, setSpeed] = useState(700); // ms per step

  const svgRef = useRef(null);

  const reset = () => {
    setRunning(false);
    setExpandedIndex(-1);
    setExpandedOrder([]);
    setFinalPath([]);
    setParents({});
  };

  const run = () => {
    reset();
    // compute GBFS once, then animate the reveal of expanded nodes
    const { expandedOrder: order, parents: cameFrom, goal: reached } = greedyBestFirst(
      start,
      goal
    );
    setExpandedOrder(order);
    setParents(cameFrom);

    setRunning(true);
    let i = -1;
    const id = setInterval(() => {
      i += 1;
      setExpandedIndex(i);
      if (i >= order.length - 1) {
        clearInterval(id);
        setRunning(false);
        const path = buildPath(cameFrom, reached ?? goal);
        setFinalPath(path);
      }
    }, Math.max(120, speed));
  };

  // Helpers to determine node/edge styles during animation
  const stepColorFor = (city) => {
    const idx = expandedOrder.indexOf(city);
    if (idx === -1 || idx > expandedIndex) return null;
    return STEP_COLORS[idx % STEP_COLORS.length];
  };

  const isOnFinalPath = (a, b = null) => {
    if (!finalPath.length) return false;
    if (b === null) return finalPath.includes(a);
    // edge on path if consecutive in finalPath
    for (let i = 0; i < finalPath.length - 1; i++) {
      if (
        (finalPath[i] === a && finalPath[i + 1] === b) ||
        (finalPath[i] === b && finalPath[i + 1] === a)
      )
        return true;
    }
    return false;
  };

  // Build list of unique undirected edges for drawing
  const uniqueEdges = useMemo(() => {
    const seen = new Set();
    const list = [];
    for (const a of Object.keys(EDGES)) {
      for (const { to: b, w } of EDGES[a]) {
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        if (!seen.has(key)) {
          seen.add(key);
          list.push({ a, b, w });
        }
      }
    }
    return list;
  }, []);

  return (
    <div className="w-full min-h-[90vh] p-4 md:p-8 bg-slate-50">
      <div className="max-w-6xl mx-auto grid md:grid-cols-[360px,1fr] gap-6">
        {/* Controls */}
        <div className="bg-white rounded-2xl shadow p-5 space-y-4 border border-slate-200">
          <h1 className="text-2xl font-semibold">Greedy Best-First Search on Romania</h1>
          <p className="text-slate-600 text-sm leading-relaxed">
            Select a <span className="font-medium">Source</span> and <span className="font-medium">Destination</span>.
            Click <span className="font-medium">SUBMIT</span> to visualize Greedy Best-First Search. Each expansion step is shown in a different color; the final solution path is highlighted in <span className="font-bold" style={{ color: "#2563eb" }}>BLUE</span>.
          </p>

          <div className="grid grid-cols-1 gap-3">
            <label className="text-sm font-medium">Source</label>
            <select
              className="rounded-xl border px-3 py-2"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              disabled={running}
            >
              {cities.map((c) => (
                <option key={c} value={c}>
                  {label(c)}
                </option>
              ))}
            </select>

            <label className="text-sm font-medium mt-2">Destination</label>
            <select
              className="rounded-xl border px-3 py-2"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              disabled={running}
            >
              {cities.map((c) => (
                <option key={c} value={c}>
                  {label(c)}
                </option>
              ))}
            </select>

            <div className="mt-2">
              <label className="text-sm font-medium">Animation speed (ms/step)</label>
              <input
                type="range"
                min={120}
                max={1500}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full"
                disabled={running}
              />
              <div className="text-xs text-slate-500">{speed} ms per step</div>
            </div>

            <div className="flex gap-2 pt-2">
              <button
                onClick={run}
                disabled={running}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-900 text-white px-4 py-2 shadow hover:shadow-md disabled:opacity-50"
                title="SUBMIT"
              >
                <Play className="w-4 h-4" /> SUBMIT
              </button>
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-200 text-slate-900 px-4 py-2 hover:bg-slate-300"
              >
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
            </div>

            <div className="mt-4 rounded-xl border p-3">
              <div className="text-sm font-medium mb-2">Legend</div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <span className="inline-flex items-center gap-2">
                  <span className="inline-block w-4 h-4 rounded-full border" style={{ background: "#e2e8f0" }} />
                  Unvisited
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="inline-block w-4 h-4 rounded-full border" style={{ background: "#fde68a" }} />
                  Frontier (pending)
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="inline-block w-4 h-4 rounded-full border" style={{ background: "#3b82f6" }} />
                  Final Path (BLUE)
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="inline-block w-4 h-4 rounded-full border bg-gradient-to-r from-[#ef4444] via-[#f59e0b] to-[#8b5cf6]" />
                  Step Colors (expansions)
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Map / Visualization */}
        <div className="bg-white rounded-2xl shadow p-4 md:p-6 border border-slate-200">
          <div className="w-full h-[72vh]">
            <svg ref={svgRef} viewBox="0 0 1000 700" className="w-full h-full">
              {/* draw edges first */}
              {uniqueEdges.map(({ a, b, w }, idx) => {
                const A = POS[a];
                const B = POS[b];
                const onFinal = isOnFinalPath(a, b);
                const expandedA = stepColorFor(a);
                const expandedB = stepColorFor(b);
                // Edge color logic: blue if on final path; else muted; if either endpoint expanded, slightly darker
                const stroke = onFinal ? "#3b82f6" : expandedA || expandedB ? "#94a3b8" : "#cbd5e1";
                const strokeWidth = onFinal ? 5 : 2.5;
                return (
                  <g key={`edge-${idx}`}>                    
                    <line
                      x1={A.x}
                      y1={A.y}
                      x2={B.x}
                      y2={B.y}
                      stroke={stroke}
                      strokeWidth={strokeWidth}
                    />
                    <text
                      x={(A.x + B.x) / 2}
                      y={(A.y + B.y) / 2 - 6}
                      fontSize={10}
                      textAnchor="middle"
                      fill="#64748b"
                    >
                      {w}
                    </text>
                  </g>
                );
              })}

              {/* draw nodes */}
              {Object.entries(POS).map(([city, { x, y }]) => {
                const onPath = isOnFinalPath(city);
                const stepColor = stepColorFor(city);
                const isSource = city === start;
                const isGoal = city === goal;

                // Determine fill
                let fill = "#e2e8f0"; // unvisited
                const wasExpanded = expandedOrder.indexOf(city) !== -1 && expandedOrder.indexOf(city) <= expandedIndex;
                if (wasExpanded && stepColor) fill = stepColor;
                if (onPath) fill = "#3b82f6"; // BLUE for final state path

                // Draw node
                return (
                  <g key={city}>
                    <circle cx={x} cy={y} r={16} fill={fill} stroke="#0f172a" strokeWidth={2} />
                    <text x={x} y={y + 30} fontSize={12} textAnchor="middle" fill="#0f172a">
                      {label(city)}
                    </text>
                    {/* source/goal badges */}
                    {isSource && (
                      <text x={x} y={y - 22} fontSize={11} textAnchor="middle" fill="#059669">SRC</text>
                    )}
                    {isGoal && (
                      <text x={x} y={y - 22} fontSize={11} textAnchor="middle" fill="#dc2626">DST</text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Status area */}
          <div className="mt-4 grid gap-2">
            <div className="text-sm text-slate-600">
              {expandedOrder.length === 0 && <span>Ready. Choose cities and press SUBMIT.</span>}
              {expandedOrder.length > 0 && expandedIndex < expandedOrder.length - 1 && (
                <span>
                  Expanded (so far): {expandedOrder.slice(0, Math.max(0, expandedIndex + 1)).map(label).join(" → ")}
                </span>
              )}
              {expandedOrder.length > 0 && expandedIndex >= expandedOrder.length - 1 && (
                <span>
                  Expansion order: {expandedOrder.map(label).join(" → ")}
                </span>
              )}
            </div>
            {finalPath.length > 0 && (
              <div className="text-sm">
                <span className="font-medium">Final path (BLUE):</span> {finalPath.map(label).join(" → ")}
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="max-w-6xl mx-auto mt-6 text-center text-xs text-slate-500">
        Heuristic: straight-line distance (Euclidean) from each node to the selected destination. Algorithm: Greedy Best‑First (expands node with smallest h(n)). Final path is shown in BLUE.
      </footer>
    </div>
  );
}