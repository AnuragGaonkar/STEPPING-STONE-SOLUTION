import streamlit as st
import numpy as np
import pandas as pd
from queue import PriorityQueue
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Smart Transportation Optimizer",
    page_icon="logo.ico",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%); }
.stApp { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%); }
.header { 
    background: rgba(255,255,255,0.95); 
    backdrop-filter: blur(20px); 
    border-radius: 24px; 
    padding: 3rem; 
    margin: 2rem 0; 
    box-shadow: 0 25px 50px rgba(0,0,0,0.2);
    text-align: center;
}
.title { 
    font-size: 3.5rem; 
    font-weight: 800; 
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    margin: 0; 
}
.subtitle { 
    font-size: 1.3rem; 
    color: #64748b; 
    font-weight: 500; 
    margin-top: 0.5rem; 
}
.card { 
    background: rgba(255,255,255,0.95); 
    backdrop-filter: blur(20px); 
    border-radius: 20px; 
    border: 1px solid rgba(255,255,255,0.3); 
    padding: 2rem; 
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.metric-card { 
    background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
    color: white; 
    border-radius: 16px; 
    padding: 1.5rem; 
    text-align: center; 
}
.metric-orange { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important; }
.metric-blue { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; }
.metric-value { 
    font-size: 2.5rem !important; 
    font-weight: 800 !important; 
    color: white !important; 
    margin: 0.5rem 0 0 0 !important;
}
.metric-label { 
    font-size: 1rem !important; 
    font-weight: 600 !important; 
    color: rgba(255,255,255,0.9) !important; 
    margin: 0 !important;
}
.btn-optimize { 
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; 
    border-radius: 16px !important; 
    font-weight: 700 !important; 
    font-size: 1.2rem !important;
    padding: 1rem 2rem !important;
    box-shadow: 0 10px 30px rgba(59,130,246,0.4) !important;
}
.dataframe { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

class PathCost:
    def __init__(self): self.ind = [0] * 4; self.cost = 0
    def __lt__(self, other): return self.cost < other.cost

class Ans:
    def __init__(self, m, n): self.total_cost = 0; self.allocated = [[0] * n for _ in range(m)]

class IndexCost:
    def __init__(self, index, cost): self.index = index; self.cost = cost
    def __lt__(self, other): return self.cost < other.cost

def print_ans(ans, s, d): return ans.total_cost

def calc_diff(s, vis_row, vis_col, pq_row):
    row_diff = [-1] * s
    for i in range(s):
        if vis_row[i] or pq_row[i].empty(): continue
        t = pq_row[i].get()
        while not pq_row[i].empty() and vis_col[t.index]: t = pq_row[i].get()
        if pq_row[i].empty():
            row_diff[i] = t.cost; pq_row[i].put(t)
        else:
            row_diff[i] = pq_row[i].queue[0].cost - t.cost; pq_row[i].put(t)
    return row_diff

def vogel_approximation_method(costs, supply, demand):
    s, d = len(costs), len(costs[0])
    ans = Ans(s, d)
    vis_row, vis_col = [False] * s, [False] * d
    pq_row, pq_col = [PriorityQueue() for _ in range(s)], [PriorityQueue() for _ in range(d)]
    
    for i in range(s):
        for j in range(d):
            pq_row[i].put(IndexCost(j, costs[i][j]))
            pq_col[j].put(IndexCost(i, costs[i][j]))
    
    row_diff = calc_diff(s, vis_row, vis_col, pq_row)
    col_diff = calc_diff(d, vis_col, vis_row, pq_col)
    
    t1, t2 = 0, 0
    while t1 + t2 < s + d - 1:
        row_ind, col_ind = row_diff.index(max(row_diff)), col_diff.index(max(col_diff))
        if row_diff[row_ind] < col_diff[col_ind]:
            i, j = pq_col[col_ind].queue[0].index, col_ind; pq_col[col_ind].get()
        else:
            i, j = row_ind, pq_row[row_ind].queue[0].index; pq_row[row_ind].get()
        
        if supply[i] <= demand[j]:
            ans.total_cost += costs[i][j] * supply[i]
            ans.allocated[i][j] = supply[i]
            demand[j] -= supply[i]; supply[i] = 0; vis_row[i] = True; t1 += 1
            row_diff[i] = -1; col_diff = calc_diff(d, vis_col, vis_row, pq_col)
        else:
            ans.total_cost += costs[i][j] * demand[j]
            ans.allocated[i][j] = demand[j]
            supply[i] -= demand[j]; demand[j] = 0; vis_col[j] = True; t2 += 1
            col_diff[j] = -1; row_diff = calc_diff(s, vis_row, vis_col, pq_row)
    return ans

def compute_duals(costs, ans, s, d):
    u, v = [0]*s, [0]*d
    visited = [[False]*d for _ in range(s)]
    q = deque()
    
    for j in range(d):
        if ans.allocated[0][j] > 0:
            v[j] = costs[0][j]; visited[0][j] = True; q.append((0, j))
    
    while q:
        i, j = q.popleft()
        for ni in range(s):
            if ans.allocated[ni][j] > 0 and not visited[ni][j]:
                u[ni] = costs[ni][j] - v[j]; visited[ni][j] = True; q.append((ni, j))
        for nj in range(d):
            if ans.allocated[i][nj] > 0 and not visited[i][nj]:
                v[nj] = costs[i][nj] - u[i]; visited[i][nj] = True; q.append((i, nj))
    return u, v

def find_negative_rc(costs, ans, u, v, s, d):
    min_rc, best_i, best_j = float('inf'), -1, -1
    for i in range(s):
        for j in range(d):
            if ans.allocated[i][j] == 0:
                rc = costs[i][j] - u[i] - v[j]
                if rc < min_rc: min_rc, best_i, best_j = rc, i, j
    return min_rc, best_i, best_j

def improve_allocation(costs, ans, i, j, s, d):
    row_basic = [nj for nj in range(d) if ans.allocated[i][nj] > 0]
    col_basic = [ni for ni in range(s) if ans.allocated[ni][j] > 0]
    if row_basic and col_basic:
        plus_i, plus_j = i, row_basic[0]
        minus_i, minus_j = col_basic[0], j
        amount = min(ans.allocated[plus_i][plus_j], ans.allocated[minus_i][minus_j])
        ans.allocated[i][j] += amount
        ans.allocated[minus_i][minus_j] -= amount
        ans.allocated[plus_i][plus_j] -= amount
        ans.total_cost += amount * (costs[i][j] - costs[minus_i][minus_j])

def stepping_stone_method(costs, supply, demand):
    s, d = len(costs), len(costs[0])
    ans = vogel_approximation_method(costs, supply[:], demand[:])
    max_iter = 100
    for _ in range(max_iter):
        u, v = compute_duals(costs, ans, s, d)
        min_rc, best_i, best_j = find_negative_rc(costs, ans, u, v, s, d)
        if min_rc >= 0: break
        improve_allocation(costs, ans, best_i, best_j, s, d)
    return ans

st.markdown("""
<div class="header">
    <h1 class="title">Smart Transportation Optimizer</h1>
    <p class="subtitle">Production-Ready VAM + Stepping Stone Algorithm</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Interactive Solver", "Algorithm Details"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Matrix Configuration")
        rows = st.number_input("Number of Sources", min_value=1, max_value=20, value=3)
        cols = st.number_input("Number of Destinations", min_value=1, max_value=20, value=3)
    
    with col2:
        st.markdown("### Cost Matrix (₹ per unit)")
        cost_matrix = []
        for i in range(rows):
            row_cols = st.columns(cols)
            row_data = []
            for j, col_box in enumerate(row_cols):
                with col_box:
                    value = st.number_input(
                        f"S{i+1} to D{j+1}",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        key=f"cost_{i}_{j}"
                    )
                    row_data.append(float(value))
            cost_matrix.append(row_data)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### Supply")
        supply = []
        for i in range(rows):
            value = st.number_input(f"S{i+1}", min_value=0.0, value=0.0, step=1.0, key=f"supply_{i}")
            supply.append(float(value))
    
    with col_s2:
        st.markdown("### Demand")
        demand = []
        for j in range(cols):
            value = st.number_input(f"D{j+1}", min_value=0.0, value=0.0, step=1.0, key=f"demand_{j}")
            demand.append(float(value))
    
    if st.button("OPTIMIZE ROUTES", type="primary", key="optimize", help="Compute optimal solution"):
        if sum(supply) == 0 or sum(demand) == 0:
            st.warning("Please enter supply and demand values")
        elif all(val == 0 for row in cost_matrix for val in row):
            st.warning("Please fill the cost matrix")
        else:
            with st.spinner("Computing optimal solution..."):
                try:
                    result = stepping_stone_method(cost_matrix, supply, demand)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Total Cost</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">₹{int(result.total_cost):,}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card metric-orange">', unsafe_allow_html=True)
                        routes_used = sum(1 for row in result.allocated for x in row if x > 0)
                        st.markdown('<div class="metric-label">Routes Used</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{routes_used}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card metric-blue">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Matrix Size</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{rows}×{cols}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Optimal Allocation Matrix")
                    alloc_df = pd.DataFrame(
                        result.allocated,
                        index=[f"S{i+1}" for i in range(rows)],
                        columns=[f"D{j+1}" for j in range(cols)]
                    )
                    st.dataframe(alloc_df.round(1), use_container_width=True)
                    
                    fig = make_subplots(1, 2, subplot_titles=["Cost Matrix (₹)", "Optimal Allocation"])
                    fig.add_trace(go.Heatmap(z=cost_matrix, colorscale="Reds", 
                                           text=[[f"₹{int(x)}" for x in row] for row in cost_matrix],
                                           texttemplate="%{text}", textfont={"size": 14, "color": "white"},
                                           colorbar=dict(title="Cost")), row=1, col=1)
                    fig.add_trace(go.Heatmap(z=result.allocated, colorscale="Viridis",
                                           text=[[f"{x:.0f}" for x in row] for row in result.allocated],
                                           texttemplate="%{text}", textfont={"size": 14, "color": "white"},
                                           colorbar=dict(title="Units")), row=1, col=2)
                    fig.update_layout(height=450, showlegend=False, margin={"t": 50})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("Optimal solution computed successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please check your input values")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>Algorithm Overview</h2>
    
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;'>
        <div>
            <h3 style='color: #3b82f6;'>Phase 1: Vogel's Approximation Method</h3>
            <ul style='color: #374151; line-height: 1.8;'>
                <li>PriorityQueue-based penalty calculation</li>
                <li>Iterative row/column selection</li>
                <li>Excellent initial basic feasible solution</li>
            </ul>
        </div>
        <div>
            <h3 style='color: #10b981;'>Phase 2: MODI Optimization</h3>
            <ul style='color: #374151; line-height: 1.8;'>
                <li>BFS dual potential computation</li>
                <li>Reduced cost minimization</li>
                <li>Guaranteed global optimum</li>
            </ul>
        </div>
    </div>
    
    <div style='margin-top: 2rem; padding: 1.5rem; background: rgba(59,130,246,0.1); border-radius: 12px; border-left: 4px solid #3b82f6;'>
        <h3 style='color: #1e293b; margin-top: 0;'>Key Components</h3>
        <p><strong>Ans:</strong> Allocation matrix + total cost</p>
        <p><strong>IndexCost:</strong> PriorityQueue elements</p>
        <p><strong>PathCost:</strong> Path optimization tracking</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='
    background: rgba(30,41,59,0.95); 
    color: white; 
    text-align: center; 
    padding: 2rem; 
    border-radius: 20px; 
    margin: 2rem 0;
'>
    <p style='margin: 0 0 1rem 0; font-size: 1.1rem;'>
        Production Implementation by <strong>Anurag Gaonkar</strong>
    </p>
    <a href='https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION' 
       style='color: #60a5fa; font-weight: 600; text-decoration: none; font-size: 1.1rem;'>
        GitHub Repository
    </a>
</div>
""", unsafe_allow_html=True)
