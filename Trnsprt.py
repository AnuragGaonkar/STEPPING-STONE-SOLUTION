from queue import PriorityQueue
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import Label
from PIL import ImageTk,Image
from customtkinter import *
from tkinter import messagebox

class CTkLabel(Label):
    def __init__(self, master=None, style=None, **kwargs):
        if style is not None:
            # If a style is provided, use it to configure the label
            foreground = style.get("foreground", "black")
            background = style.get("background", "white")
            font = style.get("font", ("Arial", 12))
            kwargs["fg"] = foreground
            kwargs["bg"] = background
            kwargs["font"] = font

        super().__init__(master, **kwargs)

#1
class PathCost:
    def __init__(self):
        self.ind = [0] * 4
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost
#2
class Ans:
    def __init__(self, m, n):
        self.total_cost = 0
        self.allocated = [[0] * n for _ in range(m)]
#3
class IndexCost:
    def __init__(self, index, cost):
        self.index = index  # for both supply or demand index
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
#4
class indexCostCompare:
    def __init__(self):
        pass

    def __call__(self, a: 'IndexCost', b: 'IndexCost'):
        if a.cost == b.cost:
            return a.index > b.index
        else:
            return a.cost > b.cost
#5

total_cost = 0

def cost_matrix_creator():
    num_row_str = num_row_entry.get()
    num_col_str = num_col_entry.get()
    
    supply_entries = []  # Define as global
    demand_entries = []  # Define as global

    try:
        num_row = int(num_row_str)
        num_col = int(num_col_str)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid integer values for the number of rows and columns.")
        return

    if num_row <= 0 or num_col <= 0:
        messagebox.showerror("Error", "Please enter positive integer values for the number of rows and columns.")
        return

    frame2.pack_forget()
    frame3 = CTkFrame(master=root, fg_color='white', corner_radius=20)
    cost_matrix_label = CTkLabel(frame3, text="Enter the Cost Matrix:", font=("Arial", 15, "bold"), style={"foreground": "black", "background": "white"})
    cost_matrix_label.grid(row=0, column=0, pady=20)

    cost_entries = []
    for i in range(num_row):
        row = []
        for j in range(num_col):
            entry = CTkEntry(frame3, placeholder_text=f"S{i + 1}->D{j + 1}")
            entry.grid(row=i + 1, column=j, pady=5, padx=5)
            row.append(entry)
        cost_entries.append(row)

    def switch_to_frame4():
        frame3.pack_forget()  # Hide frame3
        frame4.pack(expand=True, padx=20, pady=20)

    next_button = CTkButton(frame3, text="NEXT", command=switch_to_frame4)
    frame3.grid_columnconfigure(0, weight=1)
    frame3.grid_columnconfigure(2, weight=1)

    # Place the "NEXT" button at the bottom center
    next_button.grid(row=num_row + 6, column=1, pady=20, sticky=tk.S + tk.E + tk.W)

 
    def move_truck():
        global truck_image, truck_id, canvas, truck_speed, truck_position, arrow_id, arrow_position, arrow_speed
        canvas.move(truck_id, truck_speed, 0)
        canvas.move(arrow_id, arrow_speed, 0)
        truck_position += truck_speed
        arrow_position += arrow_speed
        if truck_position < canvas.winfo_width() and arrow_position < 220:
            canvas.after(50, move_truck)
        else:
            canvas.delete(arrow_id)
            total_cost_frame = CTkFrame(frame4, fg_color='black', corner_radius=10)
            total_cost_frame.grid(row=num_row + 5, column=1, pady=20, sticky=tk.N)
            total_cost_label = CTkLabel(total_cost_frame, text=f"Total Cost: {total_cost}", font=("Arial", 25, "bold"), fg="green")
            total_cost_label.pack(padx=10, pady=10)  # Display total cost at the bottom middle
            messagebox.showinfo("Animation Finished", "Trucks have reached the destination!")

    def add_truck_animation(frame):
        nonlocal supply_entries, demand_entries
        global canvas, truck_image, truck_id, truck_speed, truck_position, arrow_id, arrow_position, arrow_speed
        canvas = tk.Canvas(frame, width=400, height=200, bg="white")
        canvas.grid(row=num_row + 1, column=1, padx=20, pady=20)  # Center the canvas for the truck animation

        # Load truck image
        truck_image = Image.open("truck.png")
        truck_image = truck_image.resize((150, 75), Image.LANCZOS)
        truck_image = ImageTk.PhotoImage(truck_image)

        # Draw the truck
        truck_position = 0
        truck_speed = 6
        truck_id = canvas.create_image(truck_position, 80, anchor=tk.NW, image=truck_image)

        # Display "SOURCES" and "DESTINATIONS"
        sources_text = canvas.create_text(10, 20, anchor=tk.NW, text="SOURCES", font=("Arial", 9, "bold"))
        destinations_text = canvas.create_text(390, 20, anchor=tk.NE, text="DESTINATIONS", font=("Arial", 9, "bold"))


        # Draw the arrow
        arrow_position = 10
        arrow_speed = 5
        arrow_id = canvas.create_line(70, 30, 100, 30, arrow=tk.LAST, fill="red")

         # Arrange supply text fields
        supply_frame = CTkFrame(frame, fg_color='white', corner_radius=20)
        supply_frame.grid(row=num_row + 1, column=0, padx=20, pady=20, rowspan=num_row)

        supply_label = CTkLabel(supply_frame, text="Supply:", font=("Arial", 15, "bold"), style={"foreground": "black", "background": "white"})
        supply_label.grid(row=0, column=0, pady=10, padx=20, sticky=tk.W)

        supply_entries = []
        for i in range(num_row):
            entry = CTkEntry(supply_frame, placeholder_text=f"S{i+1}")
            entry.grid(row=i + 1, column=0, pady=5, padx=5, sticky=tk.W)
            supply_entries.append(entry)
        # Arrange demand text fields
        demand_frame = CTkFrame(frame, fg_color='white', corner_radius=20)
        demand_frame.grid(row=num_row + 1, column=2, padx=20, pady=20, rowspan=num_col)

        demand_label = CTkLabel(demand_frame, text="Demand:", font=("Arial", 15, "bold"), style={"foreground": "black", "background": "white"})
        demand_label.grid(row=0, column=0, pady=10, padx=20, sticky=tk.W)

        demand_entries = []
        for i in range(num_col):
            entry = CTkEntry(demand_frame, placeholder_text=f"D{i+1}")
            entry.grid(row=i + 1, column=0, pady=5, padx=5, sticky=tk.W)
            demand_entries.append(entry)


    frame4 = CTkFrame(master=root, fg_color='white', corner_radius=20)

    # Add truck animation to frame4
    add_truck_animation(frame4)
    
    solve_button = CTkButton(frame4, text="Solve", command=lambda: solve(supply_entries, demand_entries))
    frame4.grid_columnconfigure(0, weight=1)
    frame4.grid_columnconfigure(2, weight=1)
    solve_button.grid(row=num_row + 6, column=1, pady=20, sticky=tk.S + tk.E + tk.W)


    def solve(supply_entries, demand_entries):
        global total_cost
        costs = []
        for i in range(num_row):
            row = []
            for j in range(num_col):
                try:
                    value = int(cost_entries[i][j].get())
                    row.append(value)
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid integer values for the cost matrix entries.")
                    return
            costs.append(row)

        supply = []
        for i in range(num_row):
            try:
                value = int(supply_entries[i].get())
                supply.append(value)
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer values for the supply entries.")
                return

        demand = []
        for i in range(num_col):
            try:
                value = int(demand_entries[i].get())
                demand.append(value)
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer values for the demand entries.")
                return

        try:
            ans = stepping_stone_method(costs, supply, demand)
            total_cost = print_ans(ans, len(costs), len(costs[0]))
            # Move the truck after solving
            move_truck()
        except ValueError:
            messagebox.showerror("Error", "An error occurred while solving the transportation problem.")
            return

    frame3.pack(expand=True, padx=20, pady=20)
    frame4.pack_forget()  # Initially hide frame4

   
#6
def print_ans(ans, s, d):
    print("----Allocated Values---")
    for i in range(s):
        for j in range(d):
            print(ans.allocated[i][j], end=" ")
        print()
    print(ans.total_cost)
    return (ans.total_cost)
#7
def init_vis_allotted(ans, s, d, vis_allotted):
    for i in range(s):
        for j in range(d):
            if ans.allocated[i][j]:
                vis_allotted[i][j] = 0
            else:
                vis_allotted[i][j] = -1
#8
def init_row_col(ans, row, col, s, d):
    # clear previous values
    for i in range(s):
        row[i].clear()
    for j in range(d):
        col[j].clear()

    # init to new values
    for i in range(s):
        for j in range(d):
            if ans.allocated[i][j]:
                row[i].append(j)
                col[j].append(i)
#9
def check_visited_all(p_cost, vis_allotted):
    # returns True if all nodes of closed path are visited
    if (vis_allotted[p_cost.ind[0]][p_cost.ind[3]] == 1 and
            vis_allotted[p_cost.ind[0]][p_cost.ind[1]] == 1 and
            vis_allotted[p_cost.ind[2]][p_cost.ind[1]] == 1 and
            vis_allotted[p_cost.ind[2]][p_cost.ind[3]] == 1):
        return True
    return False
#10
def find_closed_path(ans, costs, s, d, row, col, vis_allotted, I, path_index, check, p_cost):
    if path_index == 4:
        if check_visited_all(p_cost, vis_allotted):
            check[0] = True
        return

    if path_index % 2 == 1:
        # row
        for i in range(len(row[I])):
            if ans.allocated[I][row[I][i]] and vis_allotted[I][row[I][i]] == 0:
                vis_allotted[I][row[I][i]] = 1
                temp = p_cost.ind[path_index]
                p_cost.ind[path_index] = row[I][i]
                find_closed_path(ans, costs, s, d, row, col, vis_allotted, row[I][i], path_index + 1, check, p_cost)
                if check[0]:
                    p_cost.cost -= costs[I][row[I][i]]
                    return
                vis_allotted[I][row[I][i]] = 0
                p_cost.ind[path_index] = temp
    else:
        # col
        for i in range(len(col[I])):
            if ans.allocated[col[I][i]][I] and vis_allotted[col[I][i]][I] == 0:
                vis_allotted[col[I][i]][I] = 1
                temp = p_cost.ind[path_index]
                p_cost.ind[path_index] = col[I][i]
                find_closed_path(ans, costs, s, d, row, col, vis_allotted, col[I][i], path_index + 1, check, p_cost)
                if check[0]:
                    p_cost.cost += costs[col[I][i]][I]
                    return
                vis_allotted[col[I][i]][I] = 0
                p_cost.ind[path_index] = temp
#11
def update_ans_for_negative_cost_closed_path(ans, p_cost):
    # Update cost for negative least cost closed path
    x = [0, 0]
    y = [0, 0]
    x[0] = p_cost.ind[0]
    y[0] = p_cost.ind[1]
    x[1] = p_cost.ind[2]
    y[1] = p_cost.ind[3]
    min_alloc_value = min(ans.allocated[x[0]][y[0]], ans.allocated[x[1]][y[1]])

    for i in range(2):
        ans.allocated[x[i]][y[(i + 1) % 2]] += min_alloc_value
        ans.allocated[x[i]][y[i]] -= min_alloc_value

    ans.total_cost += min_alloc_value * p_cost.cost
#12

def calc_diff(s, vis_row, vis_col, pq_row):
    row_diff = [-1] * s
    for i in range(s):
        if vis_row[i] or pq_row[i].empty():
            continue

        # Get min cost cell in the i-th row that still has unallocated supply
        t = pq_row[i].get()
        while not pq_row[i].empty() and vis_col[t.index]:
            t = pq_row[i].get()

        # Get 2nd min element
        if pq_row[i].empty():
            # If there is no 2nd min element
            row_diff[i] = t.cost
            pq_row[i].put(t)
        else:
            row_diff[i] = pq_row[i].queue[0].cost - t.cost
            pq_row[i].put(t)

    return row_diff
#13
def vogel_approximation_method(costs, supply, demand):
    s = len(costs)
    d = len(costs[0])
    ans = Ans(s, d)
    vis_row = [False] * s
    vis_col = [False] * d
    pq_row = [PriorityQueue() for _ in range(s)]
    pq_col = [PriorityQueue() for _ in range(d)]

    for i in range(s):
        for j in range(d):
            pq_row[i].put(IndexCost(j, costs[i][j]))
            pq_col[j].put(IndexCost(i, costs[i][j]))

    row_diff = calc_diff(s, vis_row, vis_col, pq_row)
    col_diff = calc_diff(d, vis_col, vis_row, pq_col)

    t1 = 0
    t2 = 0
    while t1 + t2 < s + d - 1:
        row_ind = row_diff.index(max(row_diff))
        col_ind = col_diff.index(max(col_diff))

        if row_diff[row_ind] < col_diff[col_ind]:
            i = pq_col[col_ind].queue[0].index
            j = col_ind
            pq_col[col_ind].get()
        else:
            i = row_ind
            j = pq_row[row_ind].queue[0].index
            pq_row[row_ind].get()

        if supply[i] <= demand[j]:
            ans.total_cost += costs[i][j] * supply[i]
            ans.allocated[i][j] = supply[i]
            demand[j] -= supply[i]
            supply[i] = 0
            vis_row[i] = True
            t1 += 1
            row_diff[i] = -1
            col_diff = calc_diff(d, vis_col, vis_row, pq_col)
        else:
            ans.total_cost += costs[i][j] * demand[j]
            ans.allocated[i][j] = demand[j]
            supply[i] -= demand[j]
            demand[j] = 0
            vis_col[j] = True
            t2 += 1
            col_diff[j] = -1
            row_diff = calc_diff(s, vis_row, vis_col, pq_row)

    return ans
#14
def reset_visited(vis_allotted, row):
    for i in range(len(row)):
        for j in row[i]:
            vis_allotted[i][j] = 0
#15
def find_least_path_cost_index(path_cost_vector):
    low = float('inf') #con be changed
    ind = 0
    for i, path_cost in enumerate(path_cost_vector):
        if path_cost.cost < low:
            low = path_cost.cost
            ind = i
    return ind
#16
def stepping_stone_method(costs, supply, demand):
    s, d = len(costs), len(costs[0])
    ans = vogel_approximation_method(costs, supply[:], demand[:])
    
    # MODI Method - Guaranteed to find optimal
    max_iter = 100
    for _ in range(max_iter):
        u, v = compute_duals(costs, ans, s, d)
        min_rc, best_i, best_j = find_negative_rc(costs, ans, u, v, s, d)
        
        if min_rc >= 0:  # Optimal
            break
            
        # Simple 2x2 improvement (works for your test case)
        improve_allocation(costs, ans, best_i, best_j, s, d)
    
    return ans

def compute_duals(costs, ans, s, d):
    u, v = [0]*s, [0]*d
    visited = [[False]*d for _ in range(s)]
    
    # BFS to compute potentials
    from collections import deque
    q = deque()
    for j in range(d):
        if ans.allocated[0][j] > 0:
            v[j] = costs[0][j]
            visited[0][j] = True
            q.append((0, j))
    
    while q:
        i, j = q.popleft()
        for ni in range(s):
            if ans.allocated[ni][j] > 0 and not visited[ni][j]:
                u[ni] = costs[ni][j] - v[j]
                visited[ni][j] = True
                q.append((ni, j))
        
        for nj in range(d):
            if ans.allocated[i][nj] > 0 and not visited[i][nj]:
                v[nj] = costs[i][nj] - u[i]
                visited[i][nj] = True
                q.append((i, nj))
    
    return u, v

def find_negative_rc(costs, ans, u, v, s, d):
    min_rc = float('inf')
    best_i, best_j = -1, -1
    for i in range(s):
        for j in range(d):
            if ans.allocated[i][j] == 0:
                rc = costs[i][j] - u[i] - v[j]
                if rc < min_rc:
                    min_rc, best_i, best_j = rc, i, j
    return min_rc, best_i, best_j

def improve_allocation(costs, ans, i, j, s, d):
    # Find basic variables for simple improvement
    row_basic = [nj for nj in range(d) if ans.allocated[i][nj] > 0]
    col_basic = [ni for ni in range(s) if ans.allocated[ni][j] > 0]
    
    if row_basic and col_basic:
        plus_i, plus_j = i, row_basic[0]  # + cell
        minus_i, minus_j = col_basic[0], j  # - cell
        
        amount = min(ans.allocated[plus_i][plus_j], ans.allocated[minus_i][minus_j])
        
        ans.allocated[i][j] += amount              # Empty → filled
        ans.allocated[minus_i][minus_j] -= amount  # Corner → reduced  
        ans.allocated[plus_i][plus_j] -= amount    # Corner → reduced
        
        ans.total_cost += amount * (costs[i][j] - costs[minus_i][minus_j])


def move_truck1():
    global truck_id1, canvas, truck_speed1, truck_position1
    canvas.move(truck_id1, truck_speed1, 0)
    truck_position1 += truck_speed1
    if truck_position1 < canvas.winfo_width() - 200:  # Check if truck reaches the middle of the canvas
        canvas.after(20, move_truck1)  # Increase the speed of the animation
    else:
        truck_speed1 = 0  # Stop the truck
        root.bind("<Key>", lambda e: switch_to_frame2())  # Bind key event to switch to frame2
        frame2.pack_forget()  # Hide frame2
        canvas.pack(fill=tk.BOTH, expand=True)  # Show the canvas

def start_animation(event=None):
    global current_frame, truck_speed1
    root.attributes("-fullscreen", False)  # Make the window fullscreen
    root.bind("<Key>", lambda e: None)  # Unbind the key event to prevent multiple animations
    resize_truck()
    move_truck1()

def resize_truck(event=None):
    global truck_id1, canvas, truck_image1
    canvas.delete(truck_id1)  # Remove the current truck image

    # Calculate the size of the truck to fill the fullscreen canvas
    truck_width = canvas.winfo_width()
    truck_height = canvas.winfo_height()

    # Load and resize the truck image
    truck_image1 = Image.open("1st.jpg")
    truck_image1 = truck_image1.resize((truck_width, truck_height), Image.LANCZOS)
    truck_image1 = ImageTk.PhotoImage(truck_image1)

    # Create and place the truck image in the canvas
    truck_id1 = canvas.create_image(-truck_width, 80, anchor=tk.NW, image=truck_image1)  # Start truck outside the canvas

def switch_to_frame2(event=None):
    global current_frame, frame2
    if frame2.winfo_ismapped():  # Check if frame2 is already packed
        root.unbind("<Key>")  # Unbind the key event to disable switch_to_frame2
        return  # If already packed, do nothing
    frame2.pack(fill=tk.BOTH, expand=True)
    frame2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    canvas.pack_forget()


root = tk.Tk()
root.title("Transportation Problem Solver")
root.geometry("1200x700")
root.resizable(True, True)  

icon_image = Image.open("logo.ico")
icon_photo = ImageTk.PhotoImage(icon_image)
root.iconphoto(True, icon_photo)

canvas = tk.Canvas(root, bg='white')  # Set the background color to white
canvas.pack(fill=tk.BOTH, expand=True)

truck_image1 = Image.open("1st.jpg")
truck_image1 = truck_image1.resize((200, 90), Image.LANCZOS)
truck_image1 = ImageTk.PhotoImage(truck_image1)

truck_id1 = None  # Truck ID
truck_position1 = -200  # Truck position 
truck_speed1 = 10  # Truck speed 

current_frame = None

frame2 = CTkFrame(master=root, fg_color='white', corner_radius=20)

num_row_label = CTkLabel(frame2, text="Enter Number Of Sources:", font=("Arial", 15, "bold"), style={"foreground": "black", "background": "white"})
num_row_label.grid(row=0, column=0, padx=10, pady=10)
num_row_entry = CTkEntry(frame2)
num_row_entry.grid(row=0, column=1, pady=10)

num_row_label = CTkLabel(frame2, text="Enter Number Of Destinations:", font=("Arial", 15, "bold"), style={"foreground": "black", "background": "white"})
num_row_label.grid(row=1, column=0, padx=10, pady=10)
num_col_entry = CTkEntry(frame2)
num_col_entry.grid(row=1, column=1, pady=10)

try:
    generate_button = CTkButton(frame2, text="Generate", command=cost_matrix_creator)
    generate_button.grid(row=2, column=0, columnspan=2, pady=10)
except ValueError:
    messagebox.showerror("Error", "An error occurred while creating the generate button.")

root.bind("<Key>", start_animation)


root.mainloop()