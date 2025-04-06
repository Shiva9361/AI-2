# AI Assignment 2

### S Shivadharshan : CS22B057 `Shiva9361` 
### P Akilesh : CS22B040 `Akileshdash` 
---

## Folder Structure

```
.
├── bnb_and_ida_star_on_frozen_lake.py  # Contains BnB and IDA* implementation on FrozenLake
├── tsp.py                              # Contains Hill Climbing and Simulated Annealing for TSP
├── requirements.txt                    # Required Python packages
├── TSP/                                # Submodule for TSP environment
├─── results/                           # Results folder
  ├── frozen_lake/                      # Results of frozen lake for bnb & ida*
  ├── tsp/                              # Results of tsp for hill climbing and simmulated annealing
```
#  Search Algorithms on FrozenLake & TSP

This repository implements and compares various search algorithms:

- **Branch and Bound (BnB)** and **Iterative Deepening A\*** on the **FrozenLake** environment.
- **Hill Climbing** and **Simulated Annealing** on the **Traveling Salesman Problem (TSP)** using the [`VRP-GYM`](https://github.com/kevin-schumann/vrp-gym) library.


---

## Clone the Repository

Make sure to **recursively clone** the repository to include the submodule:

```bash
git clone --recurse-submodules https://github.com/Shiva9361/AI-2.git
cd AI-2
```

If you forgot to use `--recurse-submodules`, you can still initialize the submodule manually:

```bash
git submodule init
git submodule update
```

---

## Setup Instructions

### 1. Create a Virtual Environment (Python ≥ 3.10)

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
```

---

## How to Run

### Run Branch and Bound & IDA* on FrozenLake

```bash
python3.10 bnb_and_ida_star_on_frozen_lake.py
```

### Run Hill Climbing and Simulated Annealing on TSP

```bash
python3.10 tsp.py
```

This will generate :

- GIFs of the route evolution (if `save=True` is passed)
- Plots showing execution time vs. iterations
- They will be stored in either `frozen_lake` or `tsp` folder in `results` folder present at `\`

---

## Slide Deck  : Access [here](https://docs.google.com/presentation/d/14gRg_CGgC2FAzyjdpZip9W3FMl3eKkWkqnb3jHv3bpU/edit?usp=sharing)

---

## Submodule Info

We used [kevin-schumann/vrp-gym](https://github.com/kevin-schumann/vrp-gym) as a submodule for the TSP environment simulation. Please ensure it is correctly initialized using the steps above.

---

## Requirements

- Python **3.10 or higher**
- See `requirements.txt` for all package dependencies

---

## Notes

- Both algorithms are equipped with timeout handling.
- Visualizations are generated using `matplotlib` and `imageio`.
- All runs are deterministic unless randomness is explicitly used (like in SA).
