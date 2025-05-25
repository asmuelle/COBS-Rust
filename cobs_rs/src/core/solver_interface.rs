use ndarray::{array, Array1, Array2, Axis, s};
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

use crate::core::constraints::{
    Constraint, LpConstraintRow, PointwiseConstraintKind,
    generate_monotonicity_lp_constraints, generate_convexity_lp_constraints,
    generate_pointwise_value_lp_rows, generate_pointwise_derivative_lp_rows,
    generate_boundary_lp_rows, 
    // generate_periodicity_lp_rows, // Periodicity not included for now
};
use crate::core::splines::b_spline_basis;
use crate::core::knots::validate_knots as validate_knots_for_spline;


/// Represents the solution of a COBS problem.
#[derive(Debug)]
pub struct CobsSolution {
    /// The computed B-spline coefficients.
    pub coefficients: Array1<f64>,
    // Potential future fields:
    // pub objective_value: f64,
    // pub status: SolverStatus,
    // pub slack_variables_r: Array1<f64>,
}

/// Solves the Constrained B-Spline (COBS) optimization problem.
///
/// This function formulates and solves the Linear Program (LP) to find B-spline coefficients
/// that satisfy the given constraints while minimizing an L1 data fidelity term.
/// The optimization problem is:
/// min sum(r_i)
/// s.t.
///   y_i - sum_j a_j B_j(x_i) <= r_i
/// -(y_i - sum_j a_j B_j(x_i)) <= r_i
///   r_i >= 0 for all i=0..n_data-1
///   User-defined constraints on a_j.
///
/// Variables z = [a_0, ..., a_{N-1}, r_0, ..., r_{n_data-1}]'
pub fn solve_cobs_problem(
    x_data: &Array1<f64>,
    y_data: &Array1<f64>,
    order: usize,
    knots: &Array1<f64>,
    num_coefficients: usize,
    user_constraints: &[Constraint],
    // lambda: f64, // For smoothing - future work
) -> Result<CobsSolution, String> {

    // --- 1. Input Validation ---
    if x_data.len() != y_data.len() {
        return Err("x_data and y_data must have the same length.".to_string());
    }
    if order == 0 {
        return Err("Spline order (m) must be at least 1.".to_string());
    }
    if num_coefficients == 0 && x_data.len() > 0 { // Allow num_coefficients = 0 if no data? No, spline needs coeffs.
        return Err("Number of coefficients must be at least 1.".to_string());
    }
    if num_coefficients > 0 { // Only validate knots if there are coefficients to define a spline
         validate_knots_for_spline(knots, order, num_coefficients)?;
    }

    let n_data = x_data.len();
    // If n_data is 0, the L1 fidelity part is empty. The problem is just user constraints.
    // If num_coefficients is 0 and n_data is 0, it's trivial.
    if num_coefficients == 0 && n_data == 0 {
        return Ok(CobsSolution { coefficients: Array1::zeros(0) });
    }


    // --- 2. LP Variables ---
    // z = [a_0, ..., a_{N-1}, r_0, ..., r_{n_data-1}]'
    // N = num_coefficients
    let total_vars = num_coefficients + n_data;

    // --- 3. Objective Function (min q'z) ---
    // We want to minimize sum(r_i). So, q for r_i variables is 1.0.
    // q for a_j variables is 0.0.
    let mut q_vec = vec![0.0; total_vars];
    if n_data > 0 { // Only add costs for r_i if data points exist
        for i in 0..n_data {
            q_vec[num_coefficients + i] = 1.0;
        }
    }
    
    let p_csc = CscMatrix::new(total_vars, total_vars, vec![0; total_vars + 1], Vec::new(), Vec::new());

    // --- 4. Constraint Collection ---
    // Store rows for Ax=b (ZeroCone) and Ax<=b (NonNegativeCone) separately first.
    let mut eq_rows_internal: Vec<LpConstraintRow> = Vec::new();
    let mut ineq_rows_internal: Vec<LpConstraintRow> = Vec::new();

    // Helper to pad a_row from constraint generators (which only know about `num_coefficients` vars)
    let pad_a_row_coeffs_only = |mut a_row_coeffs: Array1<f64>| -> Array1<f64> {
        if num_coefficients == 0 { return Array1::zeros(total_vars); } // No coeff variables
        let mut padded_a_row = Array1::zeros(total_vars);
        for i in 0..num_coefficients {
            padded_a_row[i] = a_row_coeffs[i];
        }
        padded_a_row
    };
    
    // Process user-defined constraints
    if num_coefficients > 0 { // User constraints apply to coefficients
        for constraint_enum in user_constraints {
            match constraint_enum {
                Constraint::Monotonicity(mc) => {
                    let (a_mat, b_vec) = generate_monotonicity_lp_constraints(num_coefficients, order, &mc.mono_type)?;
                    for i in 0..a_mat.nrows() {
                        ineq_rows_internal.push(LpConstraintRow {
                            a_row: pad_a_row_coeffs_only(a_mat.row(i).to_owned()),
                            rhs: b_vec[i],
                            kind: PointwiseConstraintKind::LessThanOrEqual, // Monotonicity generates Ax <= b
                        });
                    }
                }
                Constraint::Convexity(cc) => {
                    let (a_mat, b_vec) = generate_convexity_lp_constraints(num_coefficients, order, knots, &cc.conv_type)?;
                    for i in 0..a_mat.nrows() {
                        ineq_rows_internal.push(LpConstraintRow {
                            a_row: pad_a_row_coeffs_only(a_mat.row(i).to_owned()),
                            rhs: b_vec[i],
                            kind: PointwiseConstraintKind::LessThanOrEqual, // Convexity generates Ax <= b
                        });
                    }
                }
                Constraint::PointwiseValue(pvc) => {
                    let mut row = generate_pointwise_value_lp_rows(pvc, order, knots, num_coefficients)?;
                    row.a_row = pad_a_row_coeffs_only(row.a_row);
                    // PointwiseValueConstraint's kind determines if it's eq or ineq
                    match row.kind {
                         PointwiseConstraintKind::Equals => eq_rows_internal.push(row),
                         _ => ineq_rows_internal.push(row),
                    }
                }
                Constraint::PointwiseDerivative(pdc) => {
                    let mut row = generate_pointwise_derivative_lp_rows(pdc, order, knots, num_coefficients)?;
                    row.a_row = pad_a_row_coeffs_only(row.a_row);
                    match row.kind {
                        PointwiseConstraintKind::Equals => eq_rows_internal.push(row),
                        _ => ineq_rows_internal.push(row),
                    }
                }
                Constraint::Boundary(bc) => {
                    let mut row = generate_boundary_lp_rows(bc, order, knots, num_coefficients)?;
                    row.a_row = pad_a_row_coeffs_only(row.a_row);
                     match row.kind {
                        PointwiseConstraintKind::Equals => eq_rows_internal.push(row),
                        _ => ineq_rows_internal.push(row),
                    }
                }
                Constraint::Periodicity(_pc) => {
                    // let mut row = generate_periodicity_lp_rows(pc, order, knots, num_coefficients)?;
                    // row.a_row = pad_a_row_coeffs_only(row.a_row);
                    // eq_rows_internal.push(row);
                    // Not implementing periodicity for now as per instructions.
                    // It would be an equality constraint.
                }
            }
        }
    }


    // --- 5. L1 Data Fidelity Constraints ---
    if n_data > 0 { // Only add if there's data
        for i in 0..n_data {
            let mut b_j_xi_coeffs = Array1::zeros(num_coefficients);
            if num_coefficients > 0 {
                for j_coeff in 0..num_coefficients {
                    b_j_xi_coeffs[j_coeff] = b_spline_basis(j_coeff, order, x_data[i], knots);
                }
            }

            // Constraint 1: sum_j a_j B_j(x_i) - r_i <= y_i
            let mut a_row1 = Array1::zeros(total_vars);
            if num_coefficients > 0 {
                a_row1.slice_mut(s![0..num_coefficients]).assign(&b_j_xi_coeffs);
            }
            a_row1[num_coefficients + i] = -1.0; // for -r_i
            ineq_rows_internal.push(LpConstraintRow {
                a_row: a_row1,
                rhs: y_data[i],
                kind: PointwiseConstraintKind::LessThanOrEqual,
            });

            // Constraint 2: -sum_j a_j B_j(x_i) - r_i <= -y_i
            let mut a_row2 = Array1::zeros(total_vars);
            if num_coefficients > 0 {
                a_row2.slice_mut(s![0..num_coefficients]).assign(&(-1.0 * &b_j_xi_coeffs));
            }
            a_row2[num_coefficients + i] = -1.0; // for -r_i
            ineq_rows_internal.push(LpConstraintRow {
                a_row: a_row2,
                rhs: -y_data[i],
                kind: PointwiseConstraintKind::LessThanOrEqual,
            });
            
            // Constraint 3: -r_i <= 0 (for r_i >= 0)
            let mut a_row_r_nonneg = Array1::zeros(total_vars);
            a_row_r_nonneg[num_coefficients + i] = -1.0;
            ineq_rows_internal.push(LpConstraintRow {
                a_row: a_row_r_nonneg,
                rhs: 0.0,
                kind: PointwiseConstraintKind::LessThanOrEqual,
            });
        }
    }
    
    // --- 6. Assemble Matrices for Clarabel ---
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut b_final = Vec::new();
    let mut cones_final: Vec<SupportedConeT<f64>> = Vec::new();
    let mut current_row_idx = 0;

    // Equalities: A_eq z = b_eq  =>  A_eq z + s = b_eq, s in ZeroCone
    for row_info in &eq_rows_internal { // Iterate by reference
        // Original: a_row * z = rhs
        // Clarabel: A_c * z + s_zero = b_c. So, A_c = a_row, b_c = rhs
        for (col_idx, &val) in row_info.a_row.iter().enumerate() {
            if val.abs() > f64::EPSILON {
                a_triplets.push((current_row_idx, col_idx, val));
            }
        }
        b_final.push(row_info.rhs);
        current_row_idx += 1;
    }
    if !eq_rows_internal.is_empty() {
        cones_final.push(SupportedConeT::ZeroConeT(eq_rows_internal.len()));
    }

    // Inequalities:
    for row_info in &ineq_rows_internal { // Iterate by reference
        match row_info.kind {
            PointwiseConstraintKind::LessThanOrEqual => { // a_row * z <= rhs
                // Clarabel: A_c * z + s_nonneg = b_c. So, A_c = a_row, b_c = rhs
                for (col_idx, &val) in row_info.a_row.iter().enumerate() {
                    if val.abs() > f64::EPSILON {
                        a_triplets.push((current_row_idx, col_idx, val));
                    }
                }
                b_final.push(row_info.rhs);
            }
            PointwiseConstraintKind::GreaterThanOrEqual => { // a_row * z >= rhs  => -a_row * z <= -rhs
                // Clarabel: A_c * z + s_nonneg = b_c. Here A_c = -a_row, b_c = -rhs
                for (col_idx, &val) in row_info.a_row.iter().enumerate() {
                    if val.abs() > f64::EPSILON {
                        a_triplets.push((current_row_idx, col_idx, -val));
                    }
                }
                b_final.push(-row_info.rhs);
            }
            PointwiseConstraintKind::Equals => { // Should have been handled by eq_rows_internal
                return Err("Internal error: Equals constraint found in inequality processing queue.".to_string());
            }
        }
        current_row_idx += 1;
    }
    if !ineq_rows_internal.is_empty() { // Only add cone if there are inequality constraints
        cones_final.push(SupportedConeT::NonnegativeConeT(ineq_rows_internal.len()));
    }

    let num_total_constraints = current_row_idx;
    let a_csc = if num_total_constraints > 0 || total_vars > 0 {
        // Convert triplets to CSC format for CscMatrix::new
        let m = num_total_constraints;
        let n = total_vars;
        let nnz = a_triplets.len();

        // Calculate colptr
        let mut colptr = vec![0; n + 1];
        let mut col_counts = vec![0; n];
        for &(_, c, _) in &a_triplets {
            if c < n { // Basic bounds check for safety, though c should always be < n
                col_counts[c] += 1;
            } else {
                // This case should ideally not happen if total_vars is correct
                return Err(format!("Column index {} out of bounds for {} total variables.", c, n));
            }
        }

        colptr[0] = 0;
        for j in 0..n {
            colptr[j + 1] = colptr[j] + col_counts[j];
        }
        
        // Ensure colptr[n] == nnz, which it should be if col_counts were accurate
        if colptr[n] != nnz && nnz > 0 { // Allow nnz=0 case where colptr[n] would also be 0.
             // This might indicate an issue with col_counts accumulation or triplet generation
             return Err(format!(
                "Mismatch in CSC matrix construction: colptr[n] ({}) != nnz ({}). Col counts: {:?}, Colptr: {:?}",
                colptr[n], nnz, col_counts, colptr
            ));
        }


        // Sort triplets: column-major, then row-major
        // Clarabel's CscMatrix::new expects row indices within each column to be sorted.
        a_triplets.sort_unstable_by_key(|k| (k.1, k.0));

        let mut rowval_indices = Vec::with_capacity(nnz);
        let mut nzval_f64 = Vec::with_capacity(nnz);

        for (r, _, val) in &a_triplets {
            rowval_indices.push(*r);
            nzval_f64.push(*val);
        }
        
        // Basic validation before calling CscMatrix::new, which might panic
        if rowval_indices.len() != nnz {
            return Err(format!("Rowval length {} does not match nnz {}", rowval_indices.len(), nnz));
        }
        if nzval_f64.len() != nnz {
            return Err(format!("Nzval length {} does not match nnz {}", nzval_f64.len(), nnz));
        }
        if colptr.len() != n + 1 {
             return Err(format!("Colptr length {} does not match n+1 {}", colptr.len(), n+1));
        }


        CscMatrix::new(m, n, colptr, rowval_indices, nzval_f64)
    } else {
        // Handle case with no constraints and no variables for Clarabel
        // Or if num_total_constraints is 0 but total_vars > 0 (no rows in A)
        // Or if total_vars is 0 but num_total_constraints > 0 (no columns in A)
        if total_vars == 0 { // No columns
             CscMatrix::new(num_total_constraints, 0, vec![0], vec![], vec![])
        } else { // No rows, or both no rows and no columns (covered by first branch)
             CscMatrix::new(0, total_vars, vec![0; total_vars + 1], vec![], vec![])
        }
    };
   

    // --- 7. Solve and Return ---
    let settings = DefaultSettings::<f64>::default();

    // If no constraints and objective is trivial (e.g. all zeros if no r_i vars),
    // Clarabel might not be needed or might error.
    if total_vars == 0 { // e.g. num_coefficients = 0 and n_data = 0
        return Ok(CobsSolution { coefficients: Array1::zeros(0) });
    }
    if num_total_constraints == 0 && q_vec.iter().all(|&x| x == 0.0) {
        // No constraints, no objective on coefficients (if n_data=0). Solution is trivial a_j = 0.
         return Ok(CobsSolution { coefficients: Array1::zeros(num_coefficients) });
    }


    let mut solver = DefaultSolver::new(
        &p_csc,
        &q_vec,
        &a_csc,
        &b_final,
        &cones_final,
        settings,
    );

    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved => {
            let solution_coeffs = Array1::from_iter(solver.solution.x.iter().take(num_coefficients).cloned());
            Ok(CobsSolution {
                coefficients: solution_coeffs,
            })
        }
        _ => Err(format!("Solver did not find an optimal solution. Status: {:?}", solver.solution.status)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use crate::core::constraints::{MonotonicityConstraint, MonotonicityType, PointwiseValueConstraint, PointwiseConstraintKind};

    const TOL_SOLVER: f64 = 1e-4; // Looser tolerance for solver results

    // Helper for float array comparison with tolerance
    fn assert_array_eq_tol(a: &Array1<f64>, b: &Array1<f64>, tol: f64) {
        assert_eq!(a.len(), b.len(), "Array lengths differ. Left: {:?}, Right: {:?}", a, b);
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((val_a - val_b).abs() < tol, "Mismatch at index {}: {} vs {} (tol {})", i, val_a, val_b, tol);
        }
    }

    #[test]
    fn test_solve_cobs_simple_linear_regression() {
        // y = x
        let x_data = arr1(&[0.0, 0.5, 1.0]);
        let y_data = arr1(&[0.0, 0.5, 1.0]);
        let order = 2; // Linear spline
        let num_coefficients = 2;
        // Knots for domain [0,1], N=2, m=2. Knots len = N+m = 4.
        // Clamped: t0=t1=0, t2=t3=1 => [0,0,1,1]
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]);
        let constraints: Vec<Constraint> = Vec::new();

        let result = solve_cobs_problem(&x_data, &y_data, order, &knots, num_coefficients, &constraints);
        
        assert!(result.is_ok(), "solve_cobs_problem failed: {:?}", result.err());
        let solution = result.unwrap();
        
        assert_array_eq_tol(&solution.coefficients, &arr1(&[0.0, 1.0]), TOL_SOLVER);
    }

    #[test]
    fn test_solve_cobs_with_monotonicity() {
        let x_data = arr1(&[0.0, 0.25, 0.75, 1.0]);
        let y_data = arr1(&[0.0, 0.1,  0.9,  1.0]); 
        let order = 2;
        let num_coefficients = 3; 
        let knots = arr1(&[0.0, 0.0, 0.5, 1.0, 1.0]);
        
        let constraints = vec![
            Constraint::Monotonicity(MonotonicityConstraint {
                mono_type: MonotonicityType::Increase,
            }),
        ];

        let result = solve_cobs_problem(&x_data, &y_data, order, &knots, num_coefficients, &constraints);
        assert!(result.is_ok(), "solve_cobs_problem with monotonicity failed: {:?}", result.err());
        let solution = result.unwrap();
        
        assert!(solution.coefficients.len() == 3);
        assert!(solution.coefficients[0] <= solution.coefficients[1] + TOL_SOLVER, "a0 {} > a1 {}", solution.coefficients[0], solution.coefficients[1]);
        assert!(solution.coefficients[1] <= solution.coefficients[2] + TOL_SOLVER, "a1 {} > a2 {}", solution.coefficients[1], solution.coefficients[2]);
    }

     #[test]
    fn test_solve_cobs_with_pointwise_value_constraint() {
        let x_data = arr1(&[0.0, 0.5, 1.0]);
        let y_data = arr1(&[0.1, 0.6, 0.9]); 
        let order = 2;
        let num_coefficients = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]);
        
        let constraints = vec![
            Constraint::PointwiseValue(PointwiseValueConstraint {
                x: 0.0,
                y: 0.0,
                kind: PointwiseConstraintKind::Equals,
            }),
        ];

        let result = solve_cobs_problem(&x_data, &y_data, order, &knots, num_coefficients, &constraints);
        assert!(result.is_ok(), "solve_cobs_problem with pointwise value failed: {:?}", result.err());
        let solution = result.unwrap();

        // s(0.0) = a0 * B0(0.0) + a1 * B1(0.0)
        // B0(0.0) = 1 for these knots, B1(0.0) = 0
        // So, a0 should be 0.0
        assert!((solution.coefficients[0] - 0.0).abs() < TOL_SOLVER, "s(0.0) is not 0. a0 = {}", solution.coefficients[0]);
    }
}
