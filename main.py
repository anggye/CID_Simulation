import networkx as nx
import random
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, url_for, redirect
from io import BytesIO
from celery import Celery
from celery.result import AsyncResult
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import datetime
import time
matplotlib.use("agg")

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///simulation.db"
CORS(app)
db = SQLAlchemy(app)
app.app_context().push()


class Simulation(db.Model):
    task_id = db.Column(db.String(1000), primary_key=True)
    task_name = db.Column(db.String(100), nullable=False, default="")
    submission_time = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Integer, nullable=True, default=0)
    parameters = db.Column(db.String(1000), nullable=False)
    simulation_result = db.Column(db.JSON, nullable=True, default="")

    def __repr__(self):
        return f"Simulation('{self.task_id}', '{self.task_name}', '{self.submission_time}', '{self.duration}', '{self.parameters}', '{self.simulation_result}')"


# app.config['CELERY_BROKER_URL'] = "memory://"
# app.config['CELERY_RESULT_BACKEND'] = "cache+memory://"

# celery -A main.celery worker --loglevel=info --pool=solo

celery = Celery(
    app.name,
    broker=app.config["CELERY_BROKER_URL"],
    backend=app.config["CELERY_RESULT_BACKEND"],
)


class Node:
    # define parameters alpha and beta
    def __init__(self, type, base_alpha, base_beta, neighbors):
        self.type = type  # innovator/ordinary/conservative
        self.theta = random.uniform(0, 1)
        self.alpha = calc_alpha(base_alpha, self.type, self.theta)
        self.bias = random.uniform(
            0, 1
        )  # a normal distribution of people's bias twoards info A or B
        self.beta = calc_beta(base_beta, self.type, self.theta)
        # self.gamma = random.uniform(0.1, 0.3)
        self.timestamp_a = math.inf
        self.timestamp_b = math.inf
        self.neighbors = neighbors
        self.state = "ignorant"


def format_dateTime(time):
    today = datetime.datetime.now()
    date = today.strftime("%Y-%m-%d")
    date_time = datetime.datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M:%S")
    return date_time

def calculate_duration(submission_time):
   
    current_time = datetime.datetime.now()
    return round((current_time - submission_time).total_seconds(), 2)


def calc_alpha(base_alpha, type, theta):
    if type == "innovator":
        return min(base_alpha * (1 + theta), 1)
    elif type == "ordinary":
        return base_alpha
    elif type == "conservative":
        return base_alpha * theta


def calc_beta(base_beta, type, theta):
    if type == "innovator":
        return [min(x * (1 + theta), 1) for x in base_beta]
    elif type == "ordinary":
        return base_beta
    elif type == "conservative":
        return [x * theta for x in base_beta]


def calc_freshness(m, t, t0):
    return max(0, 1 + m * (t - t0))


@celery.task(name="main.simulate")
def simulate(params):
    if "custom_graph" in list(params.keys()):
        # custom graph has been input
        edge_list = []
        input_str = params["custom_graph"].splitlines()
        for edge in input_str:
            edge_list.append(tuple([int(t) for t in edge.split()]))
        Graph = nx.from_edgelist(edge_list)
        N = Graph.number_of_nodes()
        # graph_name = "Custom Network (N = " + str(N) + ", E = " + str(Graph.number_of_edges()) + ")"
        graph_data = {}
        graph_data["Name"] = "Custom Network"
        graph_data["Nodes"] = str(N)
        graph_data["Edges"] = str(Graph.number_of_edges())
        try:
            graph_data["Diameter"] = str(nx.diameter(Graph))
        except:
            graph_data["Diameter"] = "∞"
        graph_data["Density"] = str(np.round(nx.density(Graph), 4))
        graph_data["Average Clustering Coefficient"] = str(
            np.round(nx.average_clustering(Graph), 4)
        )
        graph_data["Number of Triangles"] = str(sum(nx.triangles(Graph).values())//3)

    #         Nodes
    # Edges
    # Diameter
    # Density
    # Average Clustering Coefficient
    # Number of triangles

    else:
        # create initial txt files for default graphs
        # generate(N, d, k, p_s, p_r)

        N = int(params["N"])  # number of nodes

        # # regular network
        # d = N//20   # degree of each node

        # # small-world network
        # k = 5  # number of nearest neighbors
        # p_s = 0.3  # probability of rewiring

        # # random network
        # p_r = 0.2  # probability of edge creation

        m = 3
        # Graphs = [nx.random_regular_graph(d, N), nx.watts_strogatz_graph(N, k, p_s), nx.erdos_renyi_graph(N, p_r)]
        Graph = nx.barabasi_albert_graph(N, m=m)
        # legend = ["Regular Network (" + str(d) + ", " + str(N) + ")", "Small-World Network" + " (" + str(k) + ", " + str(p_s) + ", " + str(N) +")", "Random Network" + " (" + str(p_r) + ", " + str(N) + ")"]
        # graph_name = "Barabási-Albert Graph (N = " + str(N) + ", m = " + str(m) + ")"
        # colors = ["b", "r", "g"]
        graph_data = {}
        graph_data["Name"] = "Barabási-Albert Graph (m = " + str(m) + ")"
        graph_data["Nodes"] = str(Graph.number_of_nodes())
        graph_data["Edges"] = str(Graph.number_of_edges())
        graph_data["Diameter"] = str(nx.diameter(Graph))
        graph_data["Density"] = str(np.round(nx.density(Graph), 4))
        graph_data["Average Clustering Coefficient"] = str(
            np.round(nx.average_clustering(Graph), 4)
        )
        graph_data["Number of Triangles"] = str(sum(nx.triangles(Graph).values()))

    n_spreaders_a = int(params["S_A"])  # number of spreaders
    n_spreaders_b = int(params["S_B"])  # number of spreaders
    base_alpha = float(params["alpha"])
    base_beta = [
        float(params["beta_A"]),
        float(params["beta_B"]),
    ]  # switching probability
    gamma = float(params["gamma"])
    n_runs = int(params["n_runs"])

    links = np.random.rand(N, N)

    # proportion of nodes
    p_innovator = float(params["p_i"]) / (
        float(params["p_i"]) + float(params["p_o"]) + float(params["p_c"])
    )
    p_ordinary = float(params["p_o"]) / (
        float(params["p_i"]) + float(params["p_o"]) + float(params["p_c"])
    )
    p_conservative = float(params["p_c"]) / (
        float(params["p_i"]) + float(params["p_o"]) + float(params["p_c"])
    )

    m = -0.01  # freshness decay slope

    # S_a_plots = []
    # S_b_plots = []
    # switcher_a_b_plots = []
    # switcher_b_a_plots = []
    # not_visited_plots = []
    # stifler_plots = []

    spreaders_runs = []
    switchers_runs = []
    not_visited_runs = []
    stifler_runs = []

    ########################
    # link strength
    # links = links/np.sum(links)
    # print(links)
    ########################
    adj_list = nx.to_dict_of_lists(Graph)
    # print(adj_list)

    for runs in range(n_runs):
        node_list = [i for i in range(N)]
        random.shuffle(node_list)

        type_mapping = {}
        for i in range(int(N * p_innovator)):
            type_mapping[node_list[i]] = "innovator"
        for i in range(int(N * p_innovator), int(N * (p_innovator + p_ordinary))):
            type_mapping[node_list[i]] = "ordinary"
        for i in range(int(N * (p_innovator + p_ordinary)), N):
            type_mapping[node_list[i]] = "conservative"

        # initialize nodes
        V = []
        for i in range(N):
            V.append(Node(type_mapping[i], base_alpha, base_beta, adj_list[i]))

        # # print parameters of each node
        # for i in range(N):
        #     print("Node " + str(i) + ": " + str(V[i].alpha) + " " + str(V[i].beta) + " " + V[i].type + " " + str(V[i].neighbors))

        S = n_spreaders_a + n_spreaders_b
        S_a = n_spreaders_a
        S_b = n_spreaders_b
        spreaders = random.sample(range(N), S)

        for i in range(S_a):
            V[spreaders[i]].state = "spreader_a"

        for i in range(S_a, S):
            V[spreaders[i]].state = "spreader_b"

        switcher_a_b = 0
        switcher_b_a = 0

        visited = []
        for temp in V:
            if temp.state[:8] == "spreader":
                visited.append(1)
            else:
                visited.append(0)

        not_visited = N - sum(visited)
        stiflers_a = 0
        stiflers_b = 0
        spreaders_graph = [[], []]
        switchers_graph = [[], []]
        not_visited_graph = []
        stiflers_graph = [[], []]

        t = 0

        while S != 0:
            not_visited = N - sum(visited)

            spreaders_graph[0].append(S_a)
            spreaders_graph[1].append(S_b)
            switchers_graph[0].append(switcher_a_b)
            switchers_graph[1].append(switcher_b_a)
            not_visited_graph.append(not_visited)
            stiflers_graph[0].append(stiflers_a)
            stiflers_graph[1].append(stiflers_b)

            for _ in range(N):
                curr = random.choice(range(N))

                interactive_neighbors = [
                    x for x in V[curr].neighbors if V[x].state[:8] == "spreader"
                ]

                if len(interactive_neighbors) == 0:
                    continue

                # from here on, node must have received some information

                elif V[curr].state == "ignorant":
                    visited[curr] = 1
                    neighbor = random.choice(interactive_neighbors)
                    if V[neighbor].state == "spreader_a":
                        V[curr].timestamp_a = t

                        prob = (
                            V[curr].alpha * links[curr][neighbor] * (1 - V[curr].bias)
                        )
                        if random.uniform(0, 1) < prob:
                            V[curr].state = "spreader_a"
                            S_a += 1
                    else:
                        V[curr].timestamp_b = t

                        prob = V[curr].alpha * links[curr][neighbor] * V[curr].bias
                        if random.uniform(0, 1) < prob:
                            V[curr].state = "spreader_b"
                            S_b += 1

                elif V[curr].state[:8] == "spreader":
                    visited[curr] = 1
                    neighbor = random.choice(interactive_neighbors)
                    if V[curr].state == V[neighbor].state:
                        # reset information freshness
                        if V[curr].state[-1] == "a":
                            V[curr].timestamp_a = t
                        else:
                            V[curr].timestamp_b = t
                    else:
                        # switching case
                        y_a = calc_freshness(m, t, V[curr].timestamp_a)
                        y_b = calc_freshness(m, t, V[curr].timestamp_b)

                        if V[curr].state[-1] == "a":
                            ##################################
                            # should the timestamp be updated here, or before calulating y_b, y_a
                            # because if we choose the latter then the conflicting information will
                            # always be fresher than the information this node is currently
                            # spreading. that doesn't seem right
                            V[curr].timestamp_b = t
                            ##################################

                            prob = min(
                                V[curr].beta[0],
                                V[curr].beta[0] * (1 + math.tanh(y_b - y_a)),
                            )
                            if random.uniform(0, 1) < prob:
                                # switch to spreader_b
                                V[curr].state = "spreader_b"
                                S_a -= 1
                                S_b += 1
                                switcher_a_b += 1

                        else:
                            V[curr].timestamp_a = t

                            prob = min(
                                V[curr].beta[1],
                                V[curr].beta[1] * (1 + math.tanh(y_a - y_b)),
                            )
                            if random.uniform(0, 1) < prob:
                                # switch to spreader_a
                                V[curr].state = "spreader_a"
                                S_b -= 1
                                S_a += 1
                                switcher_b_a += 1

            # for all spreaders, if they aren't stiflers, they become stiflers with probability gamma
            for i in range(N):
                if V[i].state[:8] == "spreader":
                    if random.uniform(0, 1) < gamma:
                        if V[i].state == "spreader_a":
                            S_a -= 1
                            stiflers_a += 1
                        elif V[i].state == "spreader_b":
                            S_b -= 1
                            stiflers_b += 1
                        V[i].state = "stifler"

            S = S_a + S_b
            t += 1

        if runs == 0:
            spreaders_runs = spreaders_graph
            switchers_runs = switchers_graph
            not_visited_runs = not_visited_graph
            stifler_runs = stiflers_graph
        else:
            # spreaders_graph end when there are no spreaders left, hence [6,5,5,4,3,0] can be extrapolated as [6,5,5,4,3,0,0,0,0,...] in case of length mismatch
            for i in range(2):  # for info A and B
                if len(spreaders_runs[i]) > len(spreaders_graph[i]):
                    for temp in range(len(spreaders_graph[i])):
                        spreaders_runs[i][temp] += spreaders_graph[i][temp]
                else:
                    for temp in range(len(spreaders_runs[i])):
                        spreaders_graph[i][temp] += spreaders_runs[i][temp]
                    spreaders_runs[i] = spreaders_graph[i]

                #################################################

                # extending and adding graph of each run to the cumulative "runs" graph
                # e.g. runs = [1,2,3,4] while graphs could be [1,2,2,2,3] (converged later for that particular run)
                # so runs should be [1,2,3,4,4] + [1,2,2,2,3] = [2,4,5,6,7]
                # this is only true for quantities which would remain constant (in the case of total spreaders becoming 0).

                if len(switchers_runs[i]) > len(switchers_graph[i]):
                    for temp in range(len(switchers_runs[i])):
                        switchers_runs[i][temp] += (
                            switchers_graph[i][temp]
                            if temp < len(switchers_graph[i])
                            else switchers_graph[i][-1]
                        )
                else:
                    for temp in range(len(switchers_graph[i])):
                        switchers_graph[i][temp] += (
                            switchers_runs[i][temp]
                            if temp < len(switchers_runs[i])
                            else switchers_runs[i][-1]
                        )
                    switchers_runs[i] = switchers_graph[i]

                if len(stifler_runs[i]) > len(stiflers_graph[i]):
                    for temp in range(len(stifler_runs[i])):
                        stifler_runs[i][temp] += (
                            stiflers_graph[i][temp]
                            if temp < len(stiflers_graph[i])
                            else stiflers_graph[i][-1]
                        )
                else:
                    for temp in range(len(stiflers_graph[i])):
                        stiflers_graph[i][temp] += (
                            stifler_runs[i][temp]
                            if temp < len(stifler_runs[i])
                            else stifler_runs[i][-1]
                        )
                    stifler_runs[i] = stiflers_graph[i]

            if len(not_visited_runs) > len(not_visited_graph):
                for temp in range(len(not_visited_runs)):
                    not_visited_runs[temp] += (
                        not_visited_graph[temp]
                        if temp < len(not_visited_graph)
                        else not_visited_graph[-1]
                    )
            else:
                for temp in range(len(not_visited_graph)):
                    not_visited_graph[temp] += (
                        not_visited_runs[temp]
                        if temp < len(not_visited_runs)
                        else not_visited_runs[-1]
                    )
                not_visited_runs = not_visited_graph

            

            # S_runs = [x + y for x, y in zip(S_runs, S_graph)]
            # S_a_runs = [x + y for x, y in zip(S_a_runs, S_a_graph)]
            # S_b_runs = [x + y for x, y in zip(S_b_runs, S_b_graph)]
            # stiflers_runs = [x + y for x, y in zip(stiflers_runs,  = [])]
            # ignorants_runs = [x + y for x, y in zip(ignorants_runs, ignorants_graph)]

    for i in range(2):
        spreaders_runs[i] = [x / n_runs for x in spreaders_runs[i]]
        switchers_runs[i] = [x / n_runs for x in switchers_runs[i]]
        stifler_runs[i] = [x / n_runs for x in stifler_runs[i]]

    not_visited_runs = [x / n_runs for x in not_visited_runs]

    ##########################################################################
    for i in range(2):
        spreaders_runs[i] = [x / N for x in spreaders_runs[i]]
        switchers_runs[i] = [x / N for x in switchers_runs[i]]
        stifler_runs[i] = [x / N for x in stifler_runs[i]]

    not_visited_runs = [x / N for x in not_visited_runs]
    ##########################################################################

    return plot_graphs(
        Graph,
        [spreaders_runs, switchers_runs, stifler_runs, not_visited_runs],
        graph_data,
    )


def plot_graphs(Graph, plots, graph_data):
    # # plot graphs in subplots, layer plots of different networks on top of each other
    # fig, axs = plt.subplots(2, 3)
    # fig.set_size_inches(10, 10)
    # for i in range(len(Graphs)):
    #     axs[0, 0].plot(S_a_plots[i], colors[i], label=legend[i])
    #     axs[0, 1].plot(S_b_plots[i], colors[i], label=legend[i])
    #     axs[0, 2].plot(switcher_a_b_plots[i], colors[i], label=legend[i])
    #     axs[1, 0].plot(switcher_b_a_plots[i], colors[i], label=legend[i])
    #     axs[1, 1].plot(not_visited_plots[i], colors[i], label=legend[i])
    #     axs[1, 2].plot(stifler_plots[i], colors[i], label=legend[i])
    # axs[0, 0].legend()
    # axs[0, 1].legend()
    # axs[0, 2].legend()
    # axs[1, 0].legend()
    # axs[1, 1].legend()
    # axs[1, 2].legend()
    # axs[0, 0].set_title("Number of spreaders of type A vs. time")
    # axs[0, 0].set(xlabel="t", ylabel="number of nodes")
    # axs[0, 1].set_title("Number of spreaders of type B vs. time")
    # axs[0, 1].set(xlabel="t", ylabel="number of nodes")
    # axs[0, 2].set_title("Number of nodes switching from A to B vs. time")
    # axs[0, 2].set(xlabel="t", ylabel="number of nodes")
    # axs[1, 0].set_title("Number of nodes switching from B to A vs. time")
    # axs[1, 0].set(xlabel="t", ylabel="number of nodes")
    # axs[1, 1].set_title("Number of nodes who didn't receive any information vs. time")
    # axs[1, 1].set(xlabel="t", ylabel="number of nodes")
    # axs[1, 2].set_title("Number of stiflers vs. time")
    # axs[1, 2].set(xlabel="t", ylabel="number of nodes")
    # fig.suptitle("Simulation Results (averaged over " + str(n_runs) + " runs per network)")

    subplot_data_list = []
    titles = [
        "Proportion of spreaders vs. time",
        "Proportion of nodes switching information vs. time",
        "Proportion of stiflers vs. time",
        "Proportion of nodes who didn't receive any information vs. time",
    ]
    plot_legends = [["Information A", "Information B"], ["A → B", "B → A"], ["A → Stifler", "B → Stifler"]]
    plot_colors = ["g", "r"]

    # fig, axs = plt.subplots()
    # nx.draw_networkx(Graph, ax=axs, node_size=40, with_labels=False)
    # axs.set_title(graph_data["Name"])
    # img_buf = BytesIO()
    # fig.savefig(img_buf, format='png')
    # img_buf.seek(0)

    # # Convert the image to base64 for embedding in HTML
    # img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    # # Close the Matplotlib figure to release resources
    # plt.close(fig)

    # Append subplot data to the list
    subplot_data_list.append({"graph_data": graph_data})

    for temp1 in range(2):
        for temp2 in range(2):
            num = temp1 * 2 + temp2
            fig, axs = plt.subplots()

            if num < 3:
                for i in range(2):
                    axs.plot(plots[num][i], plot_colors[i], label=plot_legends[num][i])
                axs.legend()
            else:
                axs.plot(plots[num], "b")

            axs.set_title(titles[num])
            axs.set(xlabel="Time", ylabel="Proportion of Nodes")
            # yticks = axs.get_yticks().tolist()
            # if 0 not in yticks:
            #     yticks = [0] + yticks
            # axs.set_yticks(yticks)

            img_buf = BytesIO()
            fig.savefig(img_buf, format="png")
            img_buf.seek(0)

            # Convert the image to base64 for embedding in HTML
            img_data = base64.b64encode(img_buf.read()).decode("utf-8")

            # Close the Matplotlib figure to release resources
            plt.close(fig)

            # Append subplot data to the list
            subplot_data_list.append({"img_data": img_data, "subplot_number": num})

    # img_buf = BytesIO()
    # plt.savefig(img_buf, format='png')
    # img_buf.seek(0)
    # plt.close()

    # # Convert the image to base64 for embedding in HTML
    # img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    return subplot_data_list


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/submit_data", methods=["POST"])
def submit_data():
    # Retrieve slider values from the request
    res = request.get_json(force=True)
    # enter a task into database
    data = res['formData']
    
    # img_data = simulate(data)
    task = simulate.apply_async(args=[data], countdown=1)
    task_instance = Simulation(
        task_id=task.id,
        task_name=res["job_name"],
        submission_time=format_dateTime(res["submission_time"]),
        duration=0,
        parameters=res["parameters"],
        simulation_result="",
    )
    db.session.add(task_instance)
    db.session.commit()
    return jsonify({}), 202, {"Location": url_for("result", task_id=task.id)}

    # if 'custom_graph' not in list(data.keys()):
    #     slider_values = list(data.values())
    #     img_data = simulate(slider_values)
    # else:
    #     # with open("./inputs/inputCustom.txt", "w") as f:
    #     #     f.write(data['custom_graph'])
    #     #     f.close()
    #     img_data = simulate()

    return jsonify(img_data)


@app.route("/result/<task_id>")
def result(task_id):
    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        return jsonify({}), 202, {"Location": url_for("result", task_id=task.id)}

    elif task.state == "SUCCESS":
        # update task in database
        task_instance = Simulation.query.filter_by(task_id=task_id).first()
        task_instance.duration = calculate_duration(task_instance.submission_time)
        task_res = task.get()
        task_instance.simulation_result = task_res
        db.session.commit()
        return jsonify({"duration": task_instance.duration}), 200
    else:
        print("task not found")
        return jsonify({}), 404, {"Location": url_for("index")}


@app.route("/result_page/<task_id>", methods=["GET"])
def result_page(task_id):
    return render_template("result.html")

@app.route("/sim_result/<task_id>", methods=["GET"])
def sim_result(task_id):
    task_instance = Simulation.query.filter_by(task_id=task_id).first()
    task_res = task_instance.simulation_result
    return jsonify(task_res)


@app.route("/about", methods=["GET"])
def get_info():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
