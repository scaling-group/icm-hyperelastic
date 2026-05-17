"""
# FEM (Finite Element Method) Code Documentation

## Overview
This FEM class is a PyTorch Lightning callback function for performing AI-based nonlinear finite element analysis.
The code combines traditional finite element methods with neural networks to predict material.

## Input Files
1. **Mesh File**: `fem/mesh/mesh.xdmf`
   - Location: fem/mesh/ folder in the project root directory
   - Format: XDMF format finite element mesh file
   - Purpose: Defines the geometry and mesh discretization of the computational domain

## Main Computational Process

### 1. Initialization Phase according to fem.yaml
- Set boundary conditions: u_displacement = 0.5 (right boundary displacement)
- Set solver parameters: 50 load steps, maximum 10 iterations per step, convergence tolerance 1e-6

### 2. Finite Element Analysis Loop
For each batch and load step:
1. **Deformation Gradient Calculation**:

2. **AI Model Prediction**:
   - Input: I1, I3 values (1711 values per mesh element)
   - Output: ψ derivatives (5 vectors)
     - ∂ψ/∂I1, ∂ψ/∂I3 (first-order derivatives)
     - ∂²ψ/∂I1², ∂²ψ/∂I1∂I3, ∂²ψ/∂I3² (second-order derivatives)

3. **Finite Element Equation Solving**:
   - Build residual equation
   - Build Jacobian matrix
   - Newton-Raphson iterative solving

### 3. Boundary Conditions
- Left boundary: Fixed displacement (0, 0)
- Right boundary: Applied displacement load (u_displacement/load_iter, 0)

## Output File Storage Locations

### 1. Main Output Directory
- Location: `{dirpath}/fem_results/`
- Contains all FEM analysis results

### 2. Displacement Field Results
- **Displacement plots**: `fem_results/all_stretches/displacement_FEM_batch_{batch_idx}_step_{step}.png`
- **Displacement data**: `fem_results/all_stretches/displacement_field_FEM_batch_{batch_idx}_step_{step}.npy`
- **Final results**: `fem_results/displacement_FEM_batch_{batch_idx}.png` (final step)

### 3. Iteration Data
- **Location**: `fem_results/iteration_data/batch_{batch_idx}_load_{load_step}/iter_{iteration}/`
- **File contents**:
  - `I.txt`: I1 and I3 values
  - `grad.txt`: ∂ψ/∂I1 and ∂ψ/∂I3 values
  - `hessian.txt`: ∂²ψ/∂I1², ∂²ψ/∂I1∂I3, ∂²ψ/∂I3² values

### 4. Log Files
- **Residual log**: `fem_results/iteration_data/residual.txt`
- Records convergence information and residual norms for each iteration

## Usage
The code acts as a PyTorch Lightning callback function, automatically triggered at the end of validation batches:
- Only processes specified batches (valid_batches_local)
- Performs complete FEM analysis for each batch
- Automatically saves all computational results and visualization images
"""

import copy
import shutil
from pathlib import Path

import fenics as fe
import imageio
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

# Disable donated buffers to avoid conflicts with gradient computation
import torch._functorch.config
import ufl
from PIL import Image

from src.utils.ice_project_utils import get_mesh_id
from src.utils.mesh_utils import load_mesh_from_dataset

torch._functorch.config.donated_buffer = False


class FEM(L.Callback):
    def __init__(
        self,
        dirpath: str,
        valid_batches_local: str,
        valid_batches_log: str,
        u_displacement: float,
        load_iter: int,
        convergence_max_iter: int,
        tol: float,
        geometry_region: str,  # Add geometry region parameter
        data_dir: str = None,  # Add data directory parameter
        dataset_type: str = None,  # Add dataset type parameter
        mesh_type: str = None,  # Add mesh type parameter
    ):
        super().__init__()
        self.dirpath = dirpath
        self.valid_batches_local = eval(valid_batches_local)
        self.valid_batches_log = eval(valid_batches_log)
        self.category = "fem"

        # FEM parameters
        self.u_displacement = u_displacement  # right boundary displacement
        self.load_iter = load_iter  # number of load steps
        self.convergence_max_iter = convergence_max_iter  # number of iterations for each load step
        self.tol = tol

        # Create output directories
        self.output_dir = Path(self.dirpath) / "fem_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "all_stretches").mkdir(exist_ok=True)
        (self.output_dir / "iteration_data").mkdir(exist_ok=True)

        self.geometry_region = eval(geometry_region)

        # Initialize residual log file
        self.residual_log_file = self.output_dir / "iteration_data" / "residual.txt"

        # Initialize list to store plot arrays for video generation
        self.plot_arrays = []

        # Store data directory and dataset type for mesh loading
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.mesh_type = mesh_type

    def log_residual(self, message):
        """
        Log message to both console and residual.txt file
        """
        print(message)
        with open(self.residual_log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def compute_grad_and_hessian(self, pl_module, raw_prompt, raw_queries):
        """
        Compute gradient and Hessian for FEM analysis
        """
        with torch.inference_mode(False):  # enable gradient calculation
            # Create a copy of raw_queries with gradient tracking
            raw_queries_with_grad = copy.deepcopy(raw_queries)

            # Set requires_grad=True for the I field in raw_queries
            raw_queries_with_grad[0] = raw_queries_with_grad[0].clone().detach().requires_grad_(True)
            target_I = raw_queries_with_grad[0]

            # Get network output with gradient tracking enabled
            network_output = pl_module.network_inference(raw_prompt=raw_prompt, raw_queries=raw_queries_with_grad)

            grad_psi_grad_Is = network_output["grad_psi_grad_Is"]

            # Calculate Hessian: ∂²ψ/∂I² with respect to raw_queries
            hessian_psi = []
            for i in range(grad_psi_grad_Is[0].shape[-1]):  # for each component of grad_psi_grad_I
                hessian_row = torch.autograd.grad(
                    outputs=grad_psi_grad_Is[0][..., i],
                    inputs=target_I,
                    grad_outputs=torch.ones_like(grad_psi_grad_Is[0][..., i]),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                hessian_psi.append(hessian_row)

            # Stack to get Hessian matrix: (bs, ..., 2, 2)
            hessian_psi = torch.stack(hessian_psi, dim=-1)  # (bs, ..., 2, 2)
            # Reshape the last two dimensions (2, 2) to (4,)
            hessian_psi = hessian_psi.reshape(*hessian_psi.shape[:-2], -1)  # (bs, ..., 4)

            # Return Hessian directly
            hessian_psi_Is = hessian_psi

        return {
            "grad_psi_grad_Is": grad_psi_grad_Is,  # PyTree
            "hessian_psi_Is": hessian_psi_Is,  # Tensor
        }

    def AI_psi(self, I1, I3, pl_module, prompt, query, batch_idx, iteration):
        """
        Get AI model predictions for psi derivatives
        """
        # Prepare modified query with current I values
        modified_query = self.prepare_modified_query_with_I(query, I1, I3)

        raw_queries = [mesh["I"] for mesh in modified_query]
        network_output = self.compute_grad_and_hessian(pl_module, prompt, raw_queries)

        # Extract specific arrays from PyTree structure
        grad_psi_grad_I = network_output["grad_psi_grad_Is"][0]  # (1, 1711, 1, 2)
        hessian_psi_Is = network_output["hessian_psi_Is"]  # (1, 1711, 1, 4)

        # Extract grad_psi_grad_I1 and grad_psi_grad_I3 from the last dimension
        grad_psi_grad_I1 = grad_psi_grad_I[0, :, 0, 0].cpu().numpy()  # (1711,)
        grad_psi_grad_I3 = grad_psi_grad_I[0, :, 0, 1].cpu().numpy()  # (1711,)

        # Extract hessian components
        grad2_psi_grad2_I1 = hessian_psi_Is[0, :, 0, 0].cpu().numpy()  # (1711,)
        grad2_psi_grad_I1_grad_I3 = (
            hessian_psi_Is[0, :, 0, 1] + hessian_psi_Is[0, :, 0, 2]
        ).cpu().numpy() / 2  # (1711,)
        grad2_psi_grad2_I3 = hessian_psi_Is[0, :, 0, 3].cpu().numpy()  # (1711,)

        return grad_psi_grad_I1, grad_psi_grad_I3, grad2_psi_grad2_I1, grad2_psi_grad_I1_grad_I3, grad2_psi_grad2_I3

    def define_boundaries(self, mesh):
        """
        Define boundary conditions from nonlinear_FEM_AI.py
        """

        # Functions and classes for boundaries
        class Left(fe.SubDomain):
            def __init__(self, geometry_region):
                super().__init__()
                self.geometry_region = geometry_region

            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], self.geometry_region[0])

        class Right(fe.SubDomain):
            def __init__(self, geometry_region):
                super().__init__()
                self.geometry_region = geometry_region

            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], self.geometry_region[1])

        left = Left(self.geometry_region)
        right = Right(self.geometry_region)

        boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        right.mark(boundaries, 1)
        return boundaries, fe.ds(subdomain_data=boundaries), left, right

    class ElementValueExpression(fe.UserExpression):
        def __init__(self, mf1, mf2, mf3, mf4, mf5, **kwargs):
            super().__init__(**kwargs)
            self.mf1 = mf1
            self.mf2 = mf2
            self.mf3 = mf3
            self.mf4 = mf4
            self.mf5 = mf5

        def eval_cell(self, value_out, x, cell):
            value_out[0] = self.mf1[cell.index]
            value_out[1] = self.mf2[cell.index]
            value_out[2] = self.mf3[cell.index]
            value_out[3] = self.mf4[cell.index]
            value_out[4] = self.mf5[cell.index]

        def value_shape(self):
            return (5,)

    def run_fem_analysis(self, pl_module, prompt, query, batch_idx, batch):
        """
        Run FEM analysis from nonlinear_FEM_AI.py logic with AI model integration
        """
        self.log_residual(f"=== Starting FEM Analysis for Batch {batch_idx} ===")

        data_dir = self.data_dir
        mesh_type = self.mesh_type

        desc = str(batch["description"][0])
        self.log_residual(f"Extracting information from batch description: {desc}")

        # Extract material_id and meshlist_ids
        material_id_str, meshlist_ids = get_mesh_id(desc)
        material = int(material_id_str)
        self.log_residual(f"Extracted material ID from description: {material}")

        # Get geometry from first mesh
        first_mesh = meshlist_ids[0]
        geometry = first_mesh[0]  # g_id is geometry
        force_type = first_mesh[1] if len(first_mesh) > 1 else "shear"
        self.log_residual(f"Extracted geometry: {geometry}, force type: {force_type}")

        # Adjust data_dir based on dataset_type from config
        # TODO: optimize this logic from directly dataset type
        if self.dataset_type == "val11":
            if self.data_dir is not None:
                data_dir = self.data_dir + "/dataset-val11" + f"-{mesh_type}"
        elif self.dataset_type == "val12D":
            if self.data_dir is not None:
                data_dir = self.data_dir + "/dataset-val12D" + f"-{mesh_type}"
        elif self.dataset_type == "val13D":
            if self.data_dir is not None:
                data_dir = self.data_dir + "/dataset-val13D" + f"-{mesh_type}"
        elif self.dataset_type == "train":
            if self.data_dir is not None:
                data_dir = self.data_dir + "/dataset-train" + f"-{mesh_type}"
        else:
            self.log_residual(f"Unknown dataset type: {self.dataset_type}")
            return

        self.log_residual("Loading mesh...")
        self.log_residual(f"Path: {data_dir}")
        self.log_residual(f"Geometry: {geometry}, Material: {material}")

        # Create mesh temporary directory in FEM output directory
        mesh_temp_dir = self.output_dir / "mesh_temp" / f"{batch_idx}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mesh_temp_dir.mkdir(parents=True, exist_ok=True)
        self.log_residual(f"Mesh XDMF temporary directory: {mesh_temp_dir}")
        stress_dir = self.output_dir / "stress"
        stress_dir.mkdir(parents=True, exist_ok=True)
        self.log_residual(f"Stress output directory: {stress_dir}")

        # Load mesh and generate XDMF file
        xdmf_file, nodes, elements = load_mesh_from_dataset(data_dir, geometry, material, force_type, mesh_temp_dir)

        self.log_residual(f"Successfully loaded mesh file: {xdmf_file}")
        self.log_residual(f"Number of nodes: {len(nodes)}, Number of elements: {len(elements)}")

        # Load mesh into FEniCS
        mesh = fe.Mesh()
        self.log_residual(f"Reading XDMF file: {xdmf_file}")
        with fe.XDMFFile(str(xdmf_file)) as infile:
            infile.read(mesh)

        # Function spaces
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        V_scalar = fe.FunctionSpace(mesh, "DG", 0)
        coords = V.tabulate_dof_coordinates()
        u_tr = fe.TrialFunction(V)
        u_test = fe.TestFunction(V)
        u = fe.Function(V)

        # Define boundaries
        boundaries, ds, left, right = self.define_boundaries(mesh)

        # Boundary conditions
        bc_load_left = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), left)
        bc_load_right = fe.DirichletBC(V, fe.Constant((self.u_displacement / float(self.load_iter), 0.0)), right)

        bc_left = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), left)
        bc_right = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), right)

        # Manual implementation of iterative solver
        du = fe.Function(V)  # Increment of the solution

        # save prompt, query, and coordinates
        self.save_fem_general_info(data_dir, geometry, material, force_type, coords, batch_idx)

        self.log_residual("Starting manual iterative solver...")
        for i in range(self.load_iter):
            for j in range(self.convergence_max_iter):
                # Calculate deformation gradient and invariants
                I_identity = fe.Identity(2)
                F = I_identity + fe.grad(u)  # deformation gradient
                C = F.T * F  # right Cauchy-Green tensor
                I1 = fe.tr(C)
                I3 = fe.det(C)

                # Project I1 and I3 to mesh functions
                I1_proj = fe.project(I1, V_scalar)
                I3_proj = fe.project(I3, V_scalar)

                # Get I1 and I3 values for all elements as vectors
                num_elements = mesh.num_cells()
                I1_values = np.zeros(num_elements)
                I3_values = np.zeros(num_elements)

                for cell in fe.cells(mesh):
                    I1_values[cell.index()] = I1_proj.vector()[cell.index()]
                    I3_values[cell.index()] = I3_proj.vector()[cell.index()]

                # Use AI model to get derivatives
                # Interface with AI model
                # input is I1_values and I3_values (two vectors with size of elements: 1711)
                # output is five vectors, psi derivatives
                (
                    grad_psi_grad_I1_vec,
                    grad_psi_grad_I3_vec,
                    grad2_psi_grad2_I1_vec,
                    grad2_psi_grad_I1_grad_I3_vec,
                    grad2_psi_grad2_I3_vec,
                ) = self.AI_psi(I1_values, I3_values, pl_module, prompt, query, batch_idx, i)

                # Save iteration data to txt file
                self.save_iteration_data(
                    I1_values,
                    I3_values,
                    grad_psi_grad_I1_vec,
                    grad_psi_grad_I3_vec,
                    grad2_psi_grad2_I1_vec,
                    grad2_psi_grad_I1_grad_I3_vec,
                    grad2_psi_grad2_I3_vec,
                    batch_idx,
                    i,
                    j,
                )

                # Initialize mesh functions for derivatives
                func_grad_psi_grad_I1 = fe.MeshFunction("double", mesh, mesh.topology().dim())
                func_grad_psi_grad_I3 = fe.MeshFunction("double", mesh, mesh.topology().dim())
                func_grad2_psi_grad2_I1 = fe.MeshFunction("double", mesh, mesh.topology().dim())
                func_grad2_psi_grad_I1_grad_I3 = fe.MeshFunction("double", mesh, mesh.topology().dim())
                func_grad2_psi_grad2_I3 = fe.MeshFunction("double", mesh, mesh.topology().dim())

                # Get the derivatives of psi for each element using analytical function
                for cell in fe.cells(mesh):
                    func_grad_psi_grad_I1[cell.index()] = grad_psi_grad_I1_vec[cell.index()]
                    func_grad_psi_grad_I3[cell.index()] = grad_psi_grad_I3_vec[cell.index()]
                    func_grad2_psi_grad2_I1[cell.index()] = grad2_psi_grad2_I1_vec[cell.index()]
                    func_grad2_psi_grad_I1_grad_I3[cell.index()] = grad2_psi_grad_I1_grad_I3_vec[cell.index()]
                    func_grad2_psi_grad2_I3[cell.index()] = grad2_psi_grad2_I3_vec[cell.index()]

                # Create expressions for derivatives
                (
                    grad_psi_grad_I1,
                    grad_psi_grad_I3,
                    grad2_psi_grad2_I1,
                    grad2_psi_grad_I1_grad_I3,
                    grad2_psi_grad2_I3,
                ) = self.ElementValueExpression(
                    func_grad_psi_grad_I1,
                    func_grad_psi_grad_I3,
                    func_grad2_psi_grad2_I1,
                    func_grad2_psi_grad_I1_grad_I3,
                    func_grad2_psi_grad2_I3,
                    degree=4,
                )

                # Analytical derivatives for FEM formulation
                grad_I1_grad_F = 2 * F
                grad_I3_grad_F = 2 * I3 * ufl.inv(F).T
                grad_psi_grad_F = grad_psi_grad_I1 * grad_I1_grad_F + grad_psi_grad_I3 * grad_I3_grad_F

                # Residual form
                Form_inner = fe.inner(grad_psi_grad_F, fe.grad(u_test)) * fe.dx
                Form_manual = Form_inner

                # Jacobian form
                grad_I1_grad_u = fe.inner(2 * F, fe.grad(u_tr))
                grad_I3_grad_u = fe.inner(2 * I3 * ufl.inv(F).T, fe.grad(u_tr))
                grad2_I1_grad_F_grad_u = 2 * fe.grad(u_tr)
                grad2_I3_grad_F_grad_u = 2 * fe.inner(2 * I3 * ufl.inv(F).T, fe.grad(u_tr)) * ufl.inv(F).T + 2 * I3 * (
                    -ufl.inv(F).T * fe.grad(u_tr).T * ufl.inv(F).T
                )

                grad2_psi_grad_I1_grad_u = (
                    grad2_psi_grad2_I1 * grad_I1_grad_u + grad2_psi_grad_I1_grad_I3 * grad_I3_grad_u
                )
                grad2_psi_grad_I3_grad_u = (
                    grad2_psi_grad2_I3 * grad_I3_grad_u + grad2_psi_grad_I1_grad_I3 * grad_I1_grad_u
                )

                grad2_psi_grad_F_grad_u = (
                    grad2_psi_grad_I1_grad_u * grad_I1_grad_F
                    + grad_psi_grad_I1 * grad2_I1_grad_F_grad_u
                    + grad2_psi_grad_I3_grad_u * grad_I3_grad_F
                    + grad_psi_grad_I3 * grad2_I3_grad_F_grad_u
                )

                Jac_inner = fe.inner(grad2_psi_grad_F_grad_u, fe.grad(u_test)) * fe.dx
                Jac_manual = Jac_inner

                # Assemble system
                A = fe.assemble(Jac_manual)
                b = fe.assemble(-Form_manual)

                # Apply boundary conditions
                if j == 0:
                    bc_load_left.apply(A, b)
                    bc_load_right.apply(A, b)
                else:
                    bc_left.apply(A, b)
                    bc_right.apply(A, b)

                # Solve linear system
                fe.solve(A, du.vector(), b)

                # Update solution
                u.vector().axpy(1.0, du.vector())

                # Check convergence
                residual_norm = b.norm("l2")
                self.log_residual(f"Load {i + 1}, Iteration {j + 1}: Residual norm = {residual_norm:.4e}")

                if residual_norm < self.tol:
                    self.log_residual("Iterative solver converged successfully!")
                    break
                elif residual_norm > self.tol and j == self.convergence_max_iter - 1:
                    self.log_residual(
                        f"Convergence tolerance not reached within {self.convergence_max_iter} iterations."
                    )

            self.log_residual(f"Load Iteration {i + 1}: Residual norm = {residual_norm:.4e}")

            # ==== compute & save Cauchy stress for this load step ====
            I_2 = fe.Identity(2)
            F = I_2 + fe.grad(u)
            C = F.T * F
            Cinv = ufl.inv(C)
            I3 = fe.det(C)

            psi_I1 = grad_psi_grad_I1
            psi_I3 = grad_psi_grad_I3

            # PK2: S = 2(ψ_I1 * I + ψ_I3 * I3 * C^{-1})
            S = 2.0 * (psi_I1 * I_2 + psi_I3 * I3 * Cinv)
            W = fe.TensorFunctionSpace(mesh, "DG", 0)
            S_proj = fe.project(S, W)

            np.save(str(stress_dir / f"PK2_batch{batch_idx}_step{i + 1}.npy"), S_proj.vector().get_local())
            # Calculate and save results
            self.save_fem_results(u, mesh, i, batch_idx)

        self.log_residual(f"=== FEM Analysis completed for Batch {batch_idx} ===")

        # Create deformation GIF animation after all load steps are completed
        self.create_deformation_video(batch_idx)

    def save_fem_results(self, u, mesh, load_step, batch_idx):
        """
        Save displacement field and create displacement visualization
        """
        # Create original plot with title and auto colorbar
        plt.figure(figsize=(8, 6))
        c1 = fe.plot(u, mode="displacement", color="black", linewidth=0.2)
        plt.colorbar(c1)
        plt.title(f"Displacement Field - Batch {batch_idx}, Step {load_step + 1}")
        plt.tight_layout()

        # Save original plot
        original_plot_file = (
            self.output_dir / "all_stretches" / f"displacement_FEM_batch_{batch_idx}_step_{load_step + 1}.png"
        )
        plt.savefig(original_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Create animation plot data in memory (not saved to disk)
        plt.figure(figsize=(8, 6), dpi=300)
        c2 = fe.plot(u, mode="displacement", color="black", linewidth=0.2)

        # Set fixed colorbar
        c2.set_clim(0, self.u_displacement)
        plt.colorbar(c2, ticks=np.linspace(0, self.u_displacement, 6))

        # Set fixed coordinate limits for consistent visualization
        plt.xlim(self.geometry_region[0] - 0.1, self.geometry_region[1] + 0.1 + self.u_displacement)
        plt.ylim(self.geometry_region[0] - 0.1, self.geometry_region[1] + 0.1)
        plt.tight_layout()

        # Convert plot to image array for GIF generation
        fig = plt.gcf()
        fig.canvas.draw()

        # Get the RGB buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf.shape = (h, w, 4)  # RGBA format

        # Convert RGBA to RGB and store in memory
        buf_rgb = buf[:, :, :3]
        self.plot_arrays.append(buf_rgb)
        plt.close()

        # Save final original plot for last step
        if load_step == self.load_iter - 1:
            final_original_plot = self.output_dir / f"displacement_FEM_batch_{batch_idx}.png"
            shutil.copy2(original_plot_file, final_original_plot)

        # Save displacement field data
        u_file = (
            self.output_dir / "all_stretches" / f"displacement_field_FEM_batch_{batch_idx}_step_{load_step + 1}.npy"
        )
        np.save(u_file, u.vector().get_local())

        if load_step == self.load_iter - 1:
            final_u_file = self.output_dir / f"displacement_field_FEM_batch_{batch_idx}.npy"
            np.save(final_u_file, u.vector().get_local())

        self.log_residual(f"Displacement field and plots saved for batch {batch_idx}, load step {load_step + 1}")

    def save_fem_general_info(self, data_dir, geometry, material, force_type, coords, batch_idx):
        """
        Save general FEM information including prompt, query, and coordinates
        """
        # Save prompt and query
        prompt_file = self.output_dir / "Query_info.txt"
        with open(prompt_file, "w") as f:
            f.write(f"Dataset: {data_dir}\n")
            f.write(f"Geometry: {geometry}\n")
            f.write(f"MaterialID: {material}\n")
            f.write(f"ForceType: {force_type}\n")

        coords_file = self.output_dir / "all_stretches" / f"coords_{batch_idx}.npy"
        np.save(coords_file, coords)

        self.log_residual(f"General FEM information saved to {self.output_dir}")

    def create_deformation_video(self, batch_idx):
        """
        Create a GIF animation showing the deformation process from plot arrays in memory
        """
        if not self.plot_arrays:
            print(f"No plot arrays found for batch {batch_idx}, skipping GIF generation")
            return

        try:
            # GIF output path
            gif_path = self.output_dir / f"deformation_animation_batch_{batch_idx}.gif"

            # Ensure all images have the same shape
            images = []
            target_shape = None

            for img_array in self.plot_arrays:
                # Set target shape from first image
                if target_shape is None:
                    target_shape = img_array.shape
                    print(f"Target image shape: {target_shape}")

                # Resize image if shape doesn't match
                if img_array.shape != target_shape:
                    pil_img = Image.fromarray(img_array)
                    pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_img)
                    print(f"Resized image from {img_array.shape} to {target_shape}")

                images.append(img_array)

            # Create GIF with 2 frames per second (adjust as needed)
            fps = 2
            duration = 1.0 / fps  # duration per frame in seconds
            imageio.mimsave(str(gif_path), images, duration=duration)

            print(f"Deformation GIF created: {gif_path}")
            print(f"GIF contains {len(images)} frames at {fps} FPS")

            # Clear the plot arrays list for next batch
            self.plot_arrays = []

        except Exception as e:
            print(f"Error creating GIF for batch {batch_idx}: {str(e)}")
            # Clear the plot arrays list even if there's an error
            self.plot_arrays = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx in self.valid_batches_local:
            # set log level for fenics solver
            # fe.set_log_level(13)

            self.log_residual(f"=== Processing Batch {batch_idx} for FEM Analysis ===")

            # Run FEM analysis with AI model integration
            self.run_fem_analysis(pl_module, batch["prpt"], batch["qury"], batch_idx, batch)

    def prepare_modified_query_with_I(self, original_query, I1, I3):
        """
        Modify specific fields of original query with given I1, I3 values
        """
        modified_query = copy.deepcopy(original_query)

        def _modify_I_field(qury_dict):
            if "I" not in qury_dict:
                return qury_dict
            I_tensor = qury_dict["I"]
            if not torch.is_tensor(I_tensor):
                I_tensor = torch.tensor(I_tensor, dtype=torch.float32)

            new_I = I_tensor.clone()

            new_I[..., 0, 0] = torch.as_tensor(I1, dtype=new_I.dtype, device=new_I.device).view(-1)
            new_I[..., 0, 1] = torch.as_tensor(I3, dtype=new_I.dtype, device=new_I.device).view(-1)
            qury_dict["I"] = new_I
            return qury_dict

        if isinstance(modified_query, list):
            for i in range(len(modified_query)):
                modified_query[i] = _modify_I_field(modified_query[i])
        elif isinstance(modified_query, dict):
            modified_query = _modify_I_field(modified_query)
        else:
            raise TypeError(f"Unexpected query type: {type(modified_query)}")

        return modified_query

    def save_iteration_data(
        self,
        I1_values,
        I3_values,
        grad_psi_grad_I1_vec,
        grad_psi_grad_I3_vec,
        grad2_psi_grad2_I1_vec,
        grad2_psi_grad_I1_grad_I3_vec,
        grad2_psi_grad2_I3_vec,
        batch_idx,
        load_step,
        iteration,
    ):
        """
        Save I, grad_psi_grad_I and hessian values to three separate txt files for each iteration
        """
        # Create iteration folder
        iter_folder_name = f"iter_{iteration + 1}"
        iter_folder = self.output_dir / "iteration_data" / f"batch_{batch_idx}_load_{load_step + 1}" / iter_folder_name
        iter_folder.mkdir(parents=True, exist_ok=True)

        # Save I values to I.txt
        I_filepath = iter_folder / "I.txt"
        with open(I_filepath, "w") as f:
            f.write(f"I1 and I3 values - Batch {batch_idx}, Load Step {load_step + 1}, Iteration {iteration + 1}\n")
            f.write("=" * 50 + "\n")
            f.write("I1\tI3\n")
            for i in range(len(I1_values)):
                f.write(f"{I1_values[i]:.6e}\t{I3_values[i]:.6e}\n")

        # Save grad_psi_grad_I values to grad.txt
        grad_filepath = iter_folder / "grad.txt"
        with open(grad_filepath, "w") as f:
            f.write(
                f"grad_psi_grad_I values - Batch {batch_idx}, Load Step {load_step + 1}, Iteration {iteration + 1}\n"
            )
            f.write("=" * 50 + "\n")
            f.write("grad_psi_grad_I1\tgrad_psi_grad_I3\n")
            for i in range(len(grad_psi_grad_I1_vec)):
                f.write(f"{grad_psi_grad_I1_vec[i]:.6e}\t{grad_psi_grad_I3_vec[i]:.6e}\n")

        # Save hessian values to hessian.txt
        hessian_filepath = iter_folder / "hessian.txt"
        with open(hessian_filepath, "w") as f:
            f.write(f"Hessian values - Batch {batch_idx}, Load Step {load_step + 1}, Iteration {iteration + 1}\n")
            f.write("=" * 50 + "\n")
            f.write("grad2_psi_grad2_I1\tgrad2_psi_grad_I1_grad_I3\tgrad2_psi_grad2_I3\n")
            for i in range(len(grad2_psi_grad2_I1_vec)):
                f.write(
                    f"{grad2_psi_grad2_I1_vec[i]:.6e}\t{grad2_psi_grad_I1_grad_I3_vec[i]:.6e}\t{grad2_psi_grad2_I3_vec[i]:.6e}\n"
                )
