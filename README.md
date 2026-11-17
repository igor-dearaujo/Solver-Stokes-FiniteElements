# Numerical Simulation of Stokes Flows via the Finite Element Method ($P^2\text{-}P^1$)

A from-scratch Python implementation of the Finite Element Method (FEM) to solve the scalar Poisson equation and the coupled incompressible Stokes equations for viscous fluid flows.

> **Academic Context:** Developed as part of the *ANN1 — Méthode des éléments finis* course in collaboration with **Igor de Araujo**.

---

## 1. Physical Understanding of the Problem

This project models **Stokes flow** (*creeping flow*), which governs viscous, incompressible fluids in the **low Reynolds number regime ($Re \ll 1$)**.

* **Viscous Dominance vs. Inertia:** Physically, when $Re \ll 1$ (such as in microfluidics, lubrication, lava flow, or microorganism locomotion), viscous forces overwhelm inertial forces by orders of magnitude. Consequently, the non-linear convective term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ becomes negligible: the flow is linear, purely dissipative, and kinematically reversible, exhibiting no turbulent wakes or inertial boundary-layer separation.
* **The Physical Role of Pressure:** Unlike compressible flows—where pressure is determined by a thermodynamic equation of state—in an incompressible fluid ($\nabla \cdot \mathbf{u} = 0$), the pressure $p$ acts as a **Lagrange multiplier**. It adjusts instantaneously across the entire domain to enforce local volume/mass conservation at every point.

---

## 2. Mathematical Formulation & Numerical Stability

### From Strong Form to Variational (Weak) Formulation
We consider the steady Stokes equations on a domain $\Omega \subset \mathbb{R}^2$:

$$\begin{cases} -\nu \Delta \mathbf{u} + \nabla p = \mathbf{f} & \text{in } \Omega \quad \text{(Momentum Conservation)} \\ \nabla \cdot \mathbf{u} = 0 & \text{in } \Omega \quad \text{(Incompressibility Constraint)} \end{cases}$$

Multiplying by test functions $(\mathbf{v}, q)$ and integrating by parts (Green's formula) yields the **weak formulation**: find $(\mathbf{u}, p) \in \mathbf{V} \times Q$ such that

$$\begin{aligned} \nu \int_{\Omega} \nabla \mathbf{u} : \nabla \mathbf{v} \, d\Omega - \int_{\Omega} p (\nabla \cdot \mathbf{v}) \, d\Omega &= \int_{\Omega} \mathbf{f} \cdot \mathbf{v} \, d\Omega \\ \int_{\Omega} q (\nabla \cdot \mathbf{u}) \, d\Omega &= 0 \end{aligned}$$

**Why this step matters:**
1. **Reduced regularity requirements:** It only requires square-integrable first derivatives ($H^1(\Omega)$) rather than continuous second derivatives, enabling the use of piecewise polynomial basis functions.
2. **Natural boundary conditions:** Stress-free outflow (Neumann) conditions arise naturally from the boundary integrals without requiring explicit nodal constraints.

### Mixed Stability and Taylor-Hood Elements ($P^2\text{-}P^1$)
When discretizing coupled velocity-pressure problems, polynomial spaces cannot be chosen independently. Using equal-order interpolation for both fields (e.g., $P^1\text{-}P^1$) violates the **LBB (*Ladyzhenskaya–Babuška–Brezzi* / *inf-sup*)** stability condition, triggering non-physical spurious pressure oscillations (*checkerboard modes*).

To guarantee stability and optimal convergence, we implemented **Taylor-Hood ($P^2\text{-}P^1$)** mixed finite elements:
* **Velocity ($\mathbf{u}_h \in P^2$):** Piecewise quadratic interpolation with **6 degrees of freedom per triangle** (3 vertices + 3 edge midpoints), providing enough kinematic richness to satisfy the divergence-free constraint.
* **Pressure ($p_h \in P^1$):** Piecewise linear continuous interpolation with **3 degrees of freedom per triangle** (vertices only).

---

## 3. Implementation Methodology

The solver was built modularly from the ground up:

1. **Scalar Validation (Poisson Equation):** Before tackling the coupled vector system, the numerical core was validated on the scalar Poisson equation with homogeneous Neumann boundary conditions. This verified the accuracy of the elemental **mass ($\mathbb{M}$)** and **stiffness ($\mathbb{K}$)** matrices using $P^1$ and $P^2$ Lagrangian basis functions.
2. **2nd-Order Mesh Generation (`Gmsh`):** Discretization of the geometric domain into quadratic triangles (`element_order=2`), extracting nodal coordinates and connectivity tables for both $P^2$ and $P^1$ spaces.
3. **Reference Element Mapping:** Symbolic (`sympy`) and numerical (`numpy`) definition of the shape functions $\hat{\varphi}_i(\xi, \eta)$ and their gradients on the reference triangle $\hat{K}$, mapped to each physical element $K$ via an affine transformation and its Jacobian matrix $J_K$.
4. **Sparse Global Assembly (`scipy.sparse`):** Assembly of the global sparse saddle-point linear system:

$$\begin{bmatrix} \nu \mathbb{K} & \mathbb{B}^T \\ \mathbb{B} & 0 \end{bmatrix} \begin{bmatrix} \mathbf{U} \\ \mathbf{P} \end{bmatrix} = \begin{bmatrix} \mathbf{F} \\ \mathbf{0} \end{bmatrix}$$

5. **Boundary Condition Enforcement:** Direct enforcement of Dirichlet boundary conditions (no-slip walls $\mathbf{u} = \mathbf{0}$ and prescribed inlet velocity profiles) into the global sparse matrix and right-hand side vector.

---

## 4. Results & Physical Analysis of the Solution

### Stage 1: Convergence Study (Poisson Benchmark)
![Poisson Convergence Plot](images/poisson_convergence.png)
* **Analysis:** Successive mesh refinement ($h \to 0$) confirmed theoretical convergence rates in the $L^2$ norm: $\mathcal{O}(h^2)$ decay for $P^1$ elements and $\mathcal{O}(h^3)$ decay for $P^2$ elements, validating the numerical quadrature and assembly of $\mathbb{M}$ and $\mathbb{K}$.

### Stage 2: $P^2\text{-}P^1$ Mesh and Velocity/Pressure Fields (Stokes Flow)
![P2-P1 Mesh](images/mesh_elements.png)
![Velocity and Pressure Fields](images/stokes_fields.png)

* **Velocity Field Behavior:** [e.g., *A smooth boundary layer develops along the no-slip walls, transitioning into a fully developed parabolic Poiseuille profile / symmetric streamlines around the obstacle.*] The symmetry of the flow field confirms the absence of convective inertial effects.
* **Pressure Field Behavior:** [e.g., *The pressure field exhibits a smooth, monotonic gradient from inlet to outlet with zero spurious oscillatory modes*], demonstrating the numerical stability of the $P^2\text{-}P^1$ Taylor-Hood discretization.

---

## 5. Repository Structure & Local Execution

```text
├── stokes_fem_solver.ipynb     # Full implementation, derivations, and visualizations
├── src/                        # Auxiliary mesh-parsing modules (Gmsh wrapper)
├── images/                     # Exported figures for documentation
└── requirements.txt            # Python environment dependencies