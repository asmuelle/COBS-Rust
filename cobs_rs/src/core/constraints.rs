//! Defines the data structures for representing various types of constraints
//! that can be applied to the B-spline curve.

/// Specifies the kind of comparison for a pointwise constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum PointwiseConstraintKind {
    /// The function (or its derivative) must be equal to the specified value.
    Equals,
    /// The function (or its derivative) must be less than or equal to the specified value.
    LessThanOrEqual,
    /// The function (or its derivative) must be greater than or equal to the specified value.
    GreaterThanOrEqual,
}

/// Specifies the type of monotonicity for a monotonicity constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum MonotonicityType {
    /// The spline function `s(x)` must be non-decreasing, i.e., `s'(x) >= 0`.
    Increase,
    /// The spline function `s(x)` must be non-increasing, i.e., `s'(x) <= 0`.
    Decrease,
}

/// Specifies the type of convexity/concavity for a convexity constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvexityType {
    /// The spline function `s(x)` must be convex, i.e., `s''(x) >= 0`.
    Convex,
    /// The spline function `s(x)` must be concave, i.e., `s''(x) <= 0`.
    Concave,
}

/// Represents a monotonicity constraint applied over the entire domain of the spline.
#[derive(Debug, Clone, PartialEq)]
pub struct MonotonicityConstraint {
    /// The type of monotonicity required (e.g., increasing or decreasing).
    pub mono_type: MonotonicityType,
}

/// Represents a convexity or concavity constraint applied over the entire domain of the spline.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexityConstraint {
    /// The type of convexity/concavity required.
    pub conv_type: ConvexityType,
}

/// Represents a constraint on the value of the spline function `s(x)` at a specific point `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointwiseValueConstraint {
    /// The point `x` at which the constraint applies.
    pub x: f64,
    /// The value `y` that `s(x)` is constrained against.
    pub y: f64,
    /// The kind of comparison (e.g., Equals, LessThanOrEqual).
    pub kind: PointwiseConstraintKind,
}

/// Represents a constraint on the value of a derivative of the spline function `s^(d)(x)` at a specific point `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointwiseDerivativeConstraint {
    /// The point `x` at which the constraint applies.
    pub x: f64,
    /// The value that the derivative is constrained against.
    pub value: f64,
    /// The order of the derivative (e.g., 1 for 1st derivative `s'(x)`, 2 for 2nd derivative `s''(x)`).
    pub derivative_order: usize,
    /// The kind of comparison (e.g., Equals, LessThanOrEqual).
    pub kind: PointwiseConstraintKind,
}

/// Represents a constraint on the function value or its derivatives at a boundary point.
/// This can be seen as a specific type of pointwise constraint, often used for initial/final conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryConstraint {
    /// The boundary point `x` (e.g., `x_min` or `x_max`).
    pub x: f64,
    /// The value `y` that the function or its derivative at `x` is constrained against.
    pub y: f64,
    /// The order of the derivative (0 for function value `s(x)`, 1 for `s'(x)`, etc.).
    pub derivative_order: usize,
    /// The kind of comparison (e.g., Equals, LessThanOrEqual).
    pub kind: PointwiseConstraintKind,
}

/// Represents a periodicity constraint, typically `g(x_start) = g(x_end)`.
/// More advanced periodicity could involve derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicityConstraint {
    /// The starting point `x_start` of the period.
    pub x_start: f64,
    /// The ending point `x_end` of the period.
    pub x_end: f64,
    // TODO: Consider adding derivative_order for periodic derivatives, e.g. g'(x_start) = g'(x_end).
    // For now, implies function value equality: g(x_start) = g(x_end).
}

/// An enum representing all possible types of constraints that can be applied to the spline.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// A monotonicity constraint (e.g., spline must be increasing).
    Monotonicity(MonotonicityConstraint),
    /// A convexity/concavity constraint (e.g., spline must be convex).
    Convexity(ConvexityConstraint),
    /// A constraint on the spline's value at a specific point.
    PointwiseValue(PointwiseValueConstraint),
    /// A constraint on a derivative of the spline at a specific point.
    PointwiseDerivative(PointwiseDerivativeConstraint),
    /// A constraint on the spline's value or derivative at a boundary.
    Boundary(BoundaryConstraint),
    /// A periodicity constraint on the spline's value between two points.
    Periodicity(PeriodicityConstraint),
}

// --- LP Constraint Generation Functions ---

use ndarray::{Array1, Array2};

/// Generates Linear Programming (LP) constraint matrices `(A, b)` for monotonicity constraints.
///
/// The constraints ensure that the B-spline coefficients `a_j` are monotonic,
/// which is a sufficient condition for the monotonicity of the spline function itself
/// under certain conditions (e.g., for order 2 and 3 splines, or generally if the
/// basis functions sum to 1 and are non-negative).
///
/// The generated constraints are of the form `A_mono * coeffs <= b_mono`.
///
/// # Arguments
/// * `num_coefficients` - The number of B-spline coefficients (N_coeffs).
/// * `_order` - The order of the B-spline (m). Currently unused as the simple
///              coefficient-based monotonicity is order-independent in this form.
/// * `constraint_type` - The type of monotonicity required (`Increase` or `Decrease`).
///
/// # Returns
/// A `Result` containing a tuple `(A_mono, b_mono)`:
/// * `A_mono`: An `Array2<f64>` where each row defines a constraint on the coefficients.
/// * `b_mono`: An `Array1<f64>` representing the right-hand side of the inequalities.
/// Returns an error string if `num_coefficients` is invalid for forming constraints.
pub fn generate_monotonicity_lp_constraints(
    num_coefficients: usize,
    _order: usize, // Marked as unused for now
    constraint_type: &MonotonicityType,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    if num_coefficients == 0 {
        // No coefficients, so no constraints possible. Return empty matrices.
        return Ok((
            Array2::zeros((0, 0)), // 0 constraints, 0 variables if num_coefficients is 0
            Array1::zeros(0),
        ));
    }
    if num_coefficients == 1 {
        // Single coefficient, no pairwise constraints. Return empty constraint matrices.
        // A matrix should have num_coefficients columns.
        return Ok((
            Array2::zeros((0, num_coefficients)), // 0 constraints
            Array1::zeros(0),
        ));
    }

    let num_constraints = num_coefficients - 1;
    let mut a_mono = Array2::zeros((num_constraints, num_coefficients));
    let b_mono = Array1::zeros(num_constraints); // b is always 0 for these constraints

    for i in 0..num_constraints {
        // Constraint involves coefficients c_i and c_{i+1}
        // Row i of A_mono corresponds to the i-th constraint.
        match constraint_type {
            MonotonicityType::Increase => {
                // We want: c[i+1] - c[i] >= 0  =>  c[i] - c[i+1] <= 0
                a_mono[[i, i]] = 1.0;
                a_mono[[i, i + 1]] = -1.0;
                // b_mono[i] is already 0.0
            }
            MonotonicityType::Decrease => {
                // We want: c[i+1] - c[i] <= 0
                a_mono[[i, i + 1]] = 1.0;
                a_mono[[i, i]] = -1.0;
                // b_mono[i] is already 0.0
            }
        }
    }

    Ok((a_mono, b_mono))
}

/// Generates Linear Programming (LP) constraint matrices `(A, b)` for convexity or concavity constraints.
///
/// The generated constraints are of the form `A_conv * coeffs <= b_conv`.
///
/// # Arguments
/// * `num_coefficients` - The number of B-spline coefficients (N_coeffs).
/// * `order` - The order of the B-spline (m). Supported orders are 2 (linear) and 3 (quadratic).
/// * `knots` - The knot vector, used for order 3 constraints.
/// * `constraint_type` - The type of convexity/concavity required (`Convex` or `Concave`).
///
/// # Returns
/// A `Result` containing a tuple `(A_conv, b_conv)`:
/// * `A_conv`: An `Array2<f64>` where each row defines a constraint.
/// * `b_conv`: An `Array1<f64>` representing the right-hand side (always zeros).
/// Returns an error string if the order is not supported or `num_coefficients` is too small.
pub fn generate_convexity_lp_constraints(
    num_coefficients: usize,
    order: usize,
    knots: &Array1<f64>,
    constraint_type: &ConvexityType,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    if order != 2 && order != 3 {
        return Err(
            "Convexity constraints are only supported for order 2 (linear) and order 3 (quadratic) splines."
                .to_string(),
        );
    }

    if num_coefficients < 3 {
        // Need at least 3 coefficients to form a second difference.
        return Ok((
            Array2::zeros((0, num_coefficients)), // 0 constraints
            Array1::zeros(0),
        ));
    }

    let num_constraints = num_coefficients - 2;
    let mut a_conv = Array2::zeros((num_constraints, num_coefficients));
    let b_conv = Array1::zeros(num_constraints); // b is always 0 for these constraints

    for j_loop_idx in 2..num_coefficients { // j_loop_idx is the index of the 'highest' coefficient in a_j, a_{j-1}, a_{j-2}
        let constraint_idx = j_loop_idx - 2; // Row index for A_conv

        if order == 2 { // Linear spline: condition on a_j - 2*a_{j-1} + a_{j-2}
            match constraint_type {
                ConvexityType::Convex => { // a_j - 2*a_{j-1} + a_{j-2} >= 0  =>  -a_{j-2} + 2*a_{j-1} - a_j <= 0
                    a_conv[[constraint_idx, j_loop_idx - 2]] = -1.0;
                    a_conv[[constraint_idx, j_loop_idx - 1]] = 2.0;
                    a_conv[[constraint_idx, j_loop_idx]] = -1.0;
                }
                ConvexityType::Concave => { // a_j - 2*a_{j-1} + a_{j-2} <= 0
                    a_conv[[constraint_idx, j_loop_idx - 2]] = 1.0;
                    a_conv[[constraint_idx, j_loop_idx - 1]] = -2.0;
                    a_conv[[constraint_idx, j_loop_idx]] = 1.0;
                }
            }
        } else { // Order == 3 (Quadratic spline)
            // Formula from paper: factor1 * (a_j - a_{j-1}) - factor2 * (a_{j-1} - a_{j-2}) >= 0 for convex
            // where j is j_loop_idx.
            // Denominators:
            // denom1 for (a_j - a_{j-1}) is (knots[j+order-1] - knots[j+1]) = (knots[j_loop_idx+3-1] - knots[j_loop_idx+1]) = (knots[j_loop_idx+2] - knots[j_loop_idx+1])
            // denom2 for (a_{j-1} - a_{j-2}) is (knots[j-1+order-1] - knots[j]) = (knots[j_loop_idx-1+3-1] - knots[j_loop_idx]) = (knots[j_loop_idx+1] - knots[j_loop_idx])

            // Check knot indices are valid. Max index for knots is num_coeffs + order - 1.
            // j_loop_idx+2 must be < knots.len().
            // knots.len() must be at least j_loop_idx+2 + 1.
            // Since j_loop_idx max is num_coeffs-1, max index needed is num_coeffs-1+2 = num_coeffs+1.
            // This must be less than knots.len(). So knots.len() > num_coeffs+1.
            // We know knots.len() = num_coeffs + order.
            // So, num_coeffs+order > num_coeffs+1 => order > 1. Which is true (order=3 here).
            // So, knots[j_loop_idx+2], knots[j_loop_idx+1], knots[j_loop_idx] are valid indices.

            let denom1 = knots[j_loop_idx + 2] - knots[j_loop_idx + 1];
            let denom2 = knots[j_loop_idx + 1] - knots[j_loop_idx];

            let factor1 = if denom1.abs() < 1e-9 { 0.0 } else { 1.0 / denom1 };
            let factor2 = if denom2.abs() < 1e-9 { 0.0 } else { 1.0 / denom2 };

            match constraint_type {
                ConvexityType::Convex => { // term1 - term2 >= 0  => term2 - term1 <= 0
                    // term2 involves a_{j-1}, a_{j-2} (coeffs at j_loop_idx-1, j_loop_idx-2)
                    // term1 involves a_j, a_{j-1} (coeffs at j_loop_idx, j_loop_idx-1)
                    a_conv[[constraint_idx, j_loop_idx - 2]] = -factor2; // Coeff of a_{j-2} from term2
                    a_conv[[constraint_idx, j_loop_idx - 1]] = factor2 + factor1; // Coeff of a_{j-1} from term2 and term1
                    a_conv[[constraint_idx, j_loop_idx    ]] = -factor1; // Coeff of a_j from term1
                }
                ConvexityType::Concave => { // term1 - term2 <= 0
                    a_conv[[constraint_idx, j_loop_idx - 2]] = factor2;
                    a_conv[[constraint_idx, j_loop_idx - 1]] = -factor2 - factor1;
                    a_conv[[constraint_idx, j_loop_idx    ]] = factor1;
                }
            }
        }
    }
    Ok((a_conv, b_conv))
}


// Helper struct for returning constraint rows
#[derive(Debug, Clone, PartialEq)]
pub struct LpConstraintRow {
    pub a_row: Array1<f64>, // Coefficients for the constraint equation: a_row * coeffs
    pub rhs: f64,           // Right-hand side of the constraint
    pub kind: PointwiseConstraintKind, // Equality, LessThanOrEqual, GreaterThanOrEqual
}

/// Generates a single LP constraint row for a pointwise value constraint `s(x) kind y`.
///
/// # Arguments
/// * `constraint` - The `PointwiseValueConstraint` definition.
/// * `order` - The order of the B-spline (m).
/// * `knots` - The knot vector.
/// * `num_coefficients` - The number of B-spline coefficients (N_coeffs).
///
/// # Returns
/// A `Result` containing an `LpConstraintRow` or an error string.
pub fn generate_pointwise_value_lp_rows(
    constraint: &PointwiseValueConstraint,
    order: usize,
    knots: &Array1<f64>,
    num_coefficients: usize,
) -> Result<LpConstraintRow, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }
    if num_coefficients == 0 {
         return Ok(LpConstraintRow {
            a_row: Array1::zeros(0),
            rhs: constraint.y,
            kind: constraint.kind.clone(),
        });
    }
    // Implicit validation of knots, order, num_coefficients via b_spline_basis calls.
    // b_spline_basis returns 0.0 for invalid j, which is correct for sum.

    let mut a_row = Array1::zeros(num_coefficients);
    for j in 0..num_coefficients {
        // b_spline_basis itself handles errors with j, order, knots internally by returning 0.0.
        // No need to use ? operator here as it returns f64 not Result.
        a_row[j] = crate::core::splines::b_spline_basis(j, order, constraint.x, knots);
    }

    Ok(LpConstraintRow {
        a_row,
        rhs: constraint.y,
        kind: constraint.kind.clone(),
    })
}

/// Generates a single LP constraint row for a pointwise derivative constraint `s^(d)(x) kind value`.
///
/// # Arguments
/// * `constraint` - The `PointwiseDerivativeConstraint` definition.
/// * `order` - The order of the B-spline (m).
/// * `knots` - The knot vector.
/// * `num_coefficients` - The number of B-spline coefficients (N_coeffs).
///
/// # Returns
/// A `Result` containing an `LpConstraintRow` or an error string.
pub fn generate_pointwise_derivative_lp_rows(
    constraint: &PointwiseDerivativeConstraint,
    order: usize,
    knots: &Array1<f64>,
    num_coefficients: usize,
) -> Result<LpConstraintRow, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }
     if constraint.derivative_order == 0 {
        return Err("Derivative order cannot be 0 for a derivative constraint. Use PointwiseValueConstraint instead.".to_string());
    }
    if constraint.derivative_order > 1 {
        // TODO: Support higher-order derivatives if needed.
        return Err("Only 1st order derivative constraints are currently supported.".to_string());
    }
     if num_coefficients == 0 {
        return Ok(LpConstraintRow {
            a_row: Array1::zeros(0),
            rhs: constraint.value,
            kind: constraint.kind.clone(),
        });
    }

    let mut a_row = Array1::zeros(num_coefficients);
    if constraint.derivative_order == 1 {
        for j in 0..num_coefficients {
            // b_spline_basis_derivative returns Result, so use ?
            a_row[j] = crate::core::splines::b_spline_basis_derivative(j, order, constraint.x, knots)?;
        }
    }
    // Else block for derivative_order > 1 would go here if supported.

    Ok(LpConstraintRow {
        a_row,
        rhs: constraint.value,
        kind: constraint.kind.clone(),
    })
}

/// Generates a single LP constraint row for a boundary constraint.
/// This function internally dispatches to value or derivative constraint generators.
pub fn generate_boundary_lp_rows(
    constraint: &BoundaryConstraint,
    order: usize,
    knots: &Array1<f64>,
    num_coefficients: usize,
) -> Result<LpConstraintRow, String> {
    if constraint.derivative_order == 0 {
        // Treat as a PointwiseValueConstraint
        let value_constraint = PointwiseValueConstraint {
            x: constraint.x,
            y: constraint.y,
            kind: constraint.kind.clone(),
        };
        generate_pointwise_value_lp_rows(&value_constraint, order, knots, num_coefficients)
    } else {
        // Treat as a PointwiseDerivativeConstraint
        let derivative_constraint = PointwiseDerivativeConstraint {
            x: constraint.x,
            value: constraint.y, // BoundaryConstraint's 'y' is the 'value' for derivative
            derivative_order: constraint.derivative_order,
            kind: constraint.kind.clone(),
        };
        generate_pointwise_derivative_lp_rows(&derivative_constraint, order, knots, num_coefficients)
    }
}

/// Generates a single LP constraint row for a periodicity constraint `s(x_start) = s(x_end)`.
///
/// The constraint implies `s(x_start) - s(x_end) = 0`.
///
/// # Arguments
/// * `constraint` - The `PeriodicityConstraint` definition.
/// * `order` - The order of the B-spline (m).
/// * `knots` - The knot vector.
/// * `num_coefficients` - The number of B-spline coefficients (N_coeffs).
///
/// # Returns
/// A `Result` containing an `LpConstraintRow` or an error string.
pub fn generate_periodicity_lp_rows(
    constraint: &PeriodicityConstraint,
    order: usize,
    knots: &Array1<f64>,
    num_coefficients: usize,
) -> Result<LpConstraintRow, String> {
    if order == 0 {
        return Err("Order must be greater than 0.".to_string());
    }
    if num_coefficients == 0 {
        // Although the loop for a_row wouldn't run, it's better to be explicit.
        // Also, Array1::zeros(0) is valid, but constraint logic implies non-empty.
        return Err("Number of coefficients must be greater than 0.".to_string());
    }
    if knots.len() != num_coefficients + order {
        return Err(format!(
            "Knots length must be equal to num_coefficients + order ({} != {} + {}).",
            knots.len(),
            num_coefficients,
            order
        ));
    }

    let mut a_row = Array1::zeros(num_coefficients);

    for j in 0..num_coefficients {
        let val_start = crate::core::splines::b_spline_basis(j, order, constraint.x_start, knots);
        let val_end = crate::core::splines::b_spline_basis(j, order, constraint.x_end, knots);
        a_row[j] = val_start - val_end;
    }

    Ok(LpConstraintRow {
        a_row,
        rhs: 0.0, // s(x_start) - s(x_end) = 0
        kind: PointwiseConstraintKind::Equals,
    })
}


#[cfg(test)]
mod tests {
    use super::*; // Import items from the parent module (constraints)
    use ndarray::{arr1, arr2}; // For convenient array creation in tests

    // Helper for float array comparison with tolerance
    fn assert_array_eq_tol(a: &Array1<f64>, b: &Array1<f64>, tol: f64) {
        assert_eq!(a.len(), b.len(), "Array lengths differ.");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((val_a - val_b).abs() < tol, "Mismatch at index {}: {} vs {} (tol {})", i, val_a, val_b, tol);
        }
    }


    // --- Tests for Monotonicity Constraints ---
    #[test]
    fn test_monotonicity_lp_constraints_increase() {
        let num_coeffs = 3;
        let order = 2; // Order is unused in current impl, but pass a typical value
        let (a, b) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Increase)
                .unwrap();

        // Expected A for Increase (c[i] - c[i+1] <= 0):
        // Row 0: c0 - c1 <= 0  => [1, -1,  0]
        // Row 1: c1 - c2 <= 0  => [0,  1, -1]
        let expected_a = arr2(&[[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]);
        let expected_b = arr1(&[0.0, 0.0]);

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_monotonicity_lp_constraints_decrease() {
        let num_coeffs = 3;
        let order = 2;
        let (a, b) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Decrease)
                .unwrap();

        // Expected A for Decrease (c[i+1] - c[i] <= 0):
        // Row 0: c1 - c0 <= 0 => [-1, 1,  0]
        // Row 1: c2 - c1 <= 0 => [ 0,-1,  1]
        let expected_a = arr2(&[[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]);
        let expected_b = arr1(&[0.0, 0.0]);

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_monotonicity_lp_constraints_num_coeffs_one() {
        let num_coeffs = 1;
        let order = 2;
        let (a, b) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Increase)
                .unwrap();

        // Expected: 0 constraints, A matrix with 1 column.
        assert_eq!(a.nrows(), 0);
        assert_eq!(a.ncols(), 1);
        assert_eq!(b.len(), 0);
    }
    
    #[test]
    fn test_monotonicity_lp_constraints_num_coeffs_zero() {
        let num_coeffs = 0;
        let order = 2;
        let (a, b) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Increase)
                .unwrap();
        
        // Expected: 0 constraints, A matrix with 0 columns.
        assert_eq!(a.nrows(), 0);
        assert_eq!(a.ncols(), 0);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_monotonicity_lp_constraints_num_coeffs_two() {
        let num_coeffs = 2;
        let order = 2;
        let (a, b) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Increase)
                .unwrap();
        
        // Expected A for Increase (c0 - c1 <= 0):
        // Row 0: [1, -1]
        let expected_a = arr2(&[[1.0, -1.0]]);
        let expected_b = arr1(&[0.0]);

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);

        let (a_dec, b_dec) =
            generate_monotonicity_lp_constraints(num_coeffs, order, &MonotonicityType::Decrease)
                .unwrap();
        // Expected A for Decrease (c1 - c0 <= 0):
        // Row 0: [-1, 1]
        let expected_a_dec = arr2(&[[-1.0, 1.0]]);
        let expected_b_dec = arr1(&[0.0]);
        assert_eq!(a_dec, expected_a_dec);
        assert_eq!(b_dec, expected_b_dec);
    }

    // --- Tests for Convexity Constraints ---

    #[test]
    fn test_convexity_lp_constraints_order2_linear_convex() {
        let num_coeffs = 3;
        let order = 2;
        // Knots are not used for order 2 in this implementation.
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0, 2.0]); // Dummy, len = N+m = 3+2=5
        let (a, b) = generate_convexity_lp_constraints(num_coeffs, order, &knots, &ConvexityType::Convex).unwrap();

        // Convex: a_j - 2*a_{j-1} + a_{j-2} >= 0 => -a_{j-2} + 2*a_{j-1} - a_j <= 0
        // j=2: -a0 + 2*a1 - a2 <= 0
        let expected_a = arr2(&[[-1.0, 2.0, -1.0]]);
        let expected_b = arr1(&[0.0]);
        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_convexity_lp_constraints_order2_linear_concave() {
        let num_coeffs = 4; // 2 constraints: a0-2a1+a2<=0, a1-2a2+a3<=0
        let order = 2;
        let knots = arr1(&[0.0; 4 + 2]); // Dummy knots
        let (a, b) = generate_convexity_lp_constraints(num_coeffs, order, &knots, &ConvexityType::Concave).unwrap();

        // Concave: a_j - 2*a_{j-1} + a_{j-2} <= 0
        // j=2:  a0 - 2*a1 + a2 <= 0  => [1, -2,  1,  0]
        // j=3:  a1 - 2*a2 + a3 <= 0  => [0,  1, -2,  1]
        let expected_a = arr2(&[[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]]);
        let expected_b = arr1(&[0.0, 0.0]);
        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_convexity_lp_constraints_order3_quadratic_convex() {
        let num_coeffs = 3; // 1 constraint
        let order = 3;
        // Knots: t0,t1,t2, t3, t4, t5 (N+m = 3+3=6)
        // Using paper formula for j=2 (constraint_idx=0): involves a0, a1, a2
        // denom1 = knots[2+3-1] - knots[2+1] = knots[4]-knots[3]
        // denom2 = knots[2-1+3-1] - knots[2] = knots[3]-knots[2]
        let knots = arr1(&[0.0, 0.0, 0.0, 1.0, 2.0, 2.0]); // Example knots, t2=0, t3=1, t4=2
        // denom1 = knots[4]-knots[3] = 2.0-1.0 = 1.0
        // denom2 = knots[3]-knots[2] = 1.0-0.0 = 1.0
        // factor1 = 1.0, factor2 = 1.0
        let (a, b) = generate_convexity_lp_constraints(num_coeffs, order, &knots, &ConvexityType::Convex).unwrap();
        
        // Convex: term2 - term1 <= 0. Where term2 uses (a1,a0), term1 uses (a2,a1)
        // A_conv[[0,0]] = -factor2 = -1.0 (for a0)
        // A_conv[[0,1]] = factor2 + factor1 = 1.0 + 1.0 = 2.0 (for a1)
        // A_conv[[0,2]] = -factor1 = -1.0 (for a2)
        let expected_a = arr2(&[[-1.0, 2.0, -1.0]]);
        let expected_b = arr1(&[0.0]);
        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }
    
    #[test]
    fn test_convexity_lp_constraints_order3_quadratic_coincident_knots() {
        let num_coeffs = 3;
        let order = 3;
        // Knots where a denominator might be zero.
        // t2=0, t3=0, t4=1. (knots[2]=0, knots[3]=0, knots[4]=1)
        // denom1 = knots[4]-knots[3] = 1-0 = 1 => factor1 = 1.0
        // denom2 = knots[3]-knots[2] = 0-0 = 0 => factor2 = 0.0
        let knots = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]); // N+m = 3+3=6
        let (a, b) = generate_convexity_lp_constraints(num_coeffs, order, &knots, &ConvexityType::Convex).unwrap();

        // Convex: term2 - term1 <= 0
        // A_conv[[0,0]] = -factor2 = 0.0
        // A_conv[[0,1]] = factor2 + factor1 = 0.0 + 1.0 = 1.0
        // A_conv[[0,2]] = -factor1 = -1.0
        let expected_a = arr2(&[[0.0, 1.0, -1.0]]);
        let expected_b = arr1(&[0.0]);
        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }


    #[test]
    fn test_convexity_lp_constraints_invalid_order() {
        let num_coeffs = 3;
        let knots = arr1(&[0.0,0.0,0.0,1.0,1.0,1.0]); // Dummy, len N+m = 3+3=6 for order 3
        assert!(generate_convexity_lp_constraints(num_coeffs, 1, &knots, &ConvexityType::Convex).is_err());
        // For order 4, knots len would be N+4 = 3+4=7
        let knots_o4 = arr1(&[0.0,0.0,0.0,0.0,1.0,1.0,1.0]);
        assert!(generate_convexity_lp_constraints(num_coeffs, 4, &knots_o4, &ConvexityType::Convex).is_err());
    }

    #[test]
    fn test_convexity_lp_constraints_too_few_coeffs() {
        let order = 2;
        let knots = arr1(&[0.0,0.0,1.0,1.0]); // N=2, m=2 -> len 4
        let (a,b) = generate_convexity_lp_constraints(2, order, &knots, &ConvexityType::Convex).unwrap();
        assert_eq!(a.nrows(), 0);
        assert_eq!(a.ncols(), 2);
        assert_eq!(b.len(), 0);

        let (a0,b0) = generate_convexity_lp_constraints(0, order, &knots, &ConvexityType::Convex).unwrap();
        assert_eq!(a0.nrows(),0);
        assert_eq!(a0.ncols(),0);
    }

    // --- Tests for Pointwise and Boundary Constraints ---
    const TOL_POINTWISE: f64 = 1e-9;

    #[test]
    fn test_generate_pointwise_value_lp_rows_simple() {
        let constraint = PointwiseValueConstraint { x: 0.5, y: 1.5, kind: PointwiseConstraintKind::Equals };
        let order = 2; // Linear
        let num_coeffs = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]); // Clamped [0,1]
        // B_0,2(0.5) for [0,0,1] = 1-0.5 = 0.5
        // B_1,2(0.5) for [0,1,1] = 0.5
        let row = generate_pointwise_value_lp_rows(&constraint, order, &knots, num_coeffs).unwrap();
        
        assert_array_eq_tol(&row.a_row, &arr1(&[0.5, 0.5]), TOL_POINTWISE);
        assert!((row.rhs - 1.5).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::Equals);
    }

    #[test]
    fn test_generate_pointwise_derivative_lp_rows_simple() {
        let constraint = PointwiseDerivativeConstraint { 
            x: 0.5, value: 2.0, derivative_order: 1, kind: PointwiseConstraintKind::LessThanOrEqual 
        };
        let order = 2; // Linear
        let num_coeffs = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]); // Clamped [0,1]
        // B'_0,2(0.5) for [0,0,1] uses B_0,1 on [0,0] (0) and B_1,1 on [0,1] (1 for x=0.5)
        // B'_0,2(0.5) = (2-1) * [ B_0,1(0.5)/(t1-t0=0) - B_1,1(0.5)/(t2-t1=1) ] = 1 * [0 - 1.0/1.0] = -1.0
        // B'_1,2(0.5) for [0,1,1] uses B_1,1 on [0,1] (1) and B_2,1 on [1,1] (0)
        // B'_1,2(0.5) = (2-1) * [ B_1,1(0.5)/(t2-t1=1) - B_2,1(0.5)/(t3-t2=0) ] = 1 * [1.0/1.0 - 0] = 1.0
        let row = generate_pointwise_derivative_lp_rows(&constraint, order, &knots, num_coeffs).unwrap();

        assert_array_eq_tol(&row.a_row, &arr1(&[-1.0, 1.0]), TOL_POINTWISE);
        assert!((row.rhs - 2.0).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::LessThanOrEqual);
    }
    
    #[test]
    fn test_generate_boundary_lp_rows_value() {
        let constraint = BoundaryConstraint { 
            x: 0.0, y: 1.0, derivative_order: 0, kind: PointwiseConstraintKind::Equals 
        };
        let order = 2;
        let num_coeffs = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]); // Clamped [0,1]
        // B_0,2(0.0) = 1, B_1,2(0.0) = 0
        let row = generate_boundary_lp_rows(&constraint, order, &knots, num_coeffs).unwrap();
        assert_array_eq_tol(&row.a_row, &arr1(&[1.0, 0.0]), TOL_POINTWISE);
        assert!((row.rhs - 1.0).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::Equals);
    }

    #[test]
    fn test_generate_boundary_lp_rows_derivative() {
        let constraint = BoundaryConstraint { 
            x: 1.0, y: -1.0, derivative_order: 1, kind: PointwiseConstraintKind::GreaterThanOrEqual
        };
        let order = 2;
        let num_coeffs = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]); // Clamped [0,1]
        // B'_0,2(1.0) = -1.0 (deriv of 1-x)
        // B'_1,2(1.0) =  1.0 (deriv of x)
        let row = generate_boundary_lp_rows(&constraint, order, &knots, num_coeffs).unwrap();
        assert_array_eq_tol(&row.a_row, &arr1(&[-1.0, 1.0]), TOL_POINTWISE);
        assert!((row.rhs - (-1.0)).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::GreaterThanOrEqual);
    }

    #[test]
    fn test_pointwise_derivative_unsupported_order() {
        let constraint = PointwiseDerivativeConstraint { 
            x: 0.5, value: 2.0, derivative_order: 2, kind: PointwiseConstraintKind::Equals 
        };
        let order = 3;
        let num_coeffs = 3;
        let knots = arr1(&[0.0,0.0,0.0,1.0,1.0,1.0]);
        assert!(generate_pointwise_derivative_lp_rows(&constraint, order, &knots, num_coeffs).is_err());
    }
    
    #[test]
    fn test_pointwise_value_order_zero() {
        let constraint = PointwiseValueConstraint { x: 0.5, y: 1.5, kind: PointwiseConstraintKind::Equals };
        assert!(generate_pointwise_value_lp_rows(&constraint, 0, &arr1(&[0.,0.,1.,1.]), 2).is_err());
    }

    #[test]
    fn test_pointwise_derivative_order_zero() {
         let constraint = PointwiseDerivativeConstraint { 
            x: 0.5, value: 2.0, derivative_order: 1, kind: PointwiseConstraintKind::Equals 
        };
        assert!(generate_pointwise_derivative_lp_rows(&constraint, 0, &arr1(&[0.,0.,1.,1.]), 2).is_err());
    }
     #[test]
    fn test_pointwise_derivative_deriv_order_zero() {
         let constraint = PointwiseDerivativeConstraint { 
            x: 0.5, value: 2.0, derivative_order: 0, kind: PointwiseConstraintKind::Equals 
        };
        assert!(generate_pointwise_derivative_lp_rows(&constraint, 2, &arr1(&[0.,0.,1.,1.]), 2).is_err());
    }

    // --- Tests for Periodicity Constraints ---

    #[test]
    fn test_generate_periodicity_lp_rows_simple() {
        let constraint = PeriodicityConstraint { x_start: 0.0, x_end: 1.0 };
        let order = 2; // Linear
        let num_coeffs = 2;
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]); // Clamped [0,1], N+m = 2+2=4
        // B_0,2(0.0) = 1.0, B_0,2(1.0) = 0.0 => a_row[0] = 1.0 - 0.0 = 1.0
        // B_1,2(0.0) = 0.0, B_1,2(1.0) = 1.0 => a_row[1] = 0.0 - 1.0 = -1.0
        let row = generate_periodicity_lp_rows(&constraint, order, &knots, num_coeffs).unwrap();
        
        assert_array_eq_tol(&row.a_row, &arr1(&[1.0, -1.0]), TOL_POINTWISE);
        assert!((row.rhs - 0.0).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::Equals);
    }

    #[test]
    fn test_generate_periodicity_lp_rows_cubic_example() {
        // Example from a practical scenario: s(x_min) = s(x_max)
        // Let domain be [0, 3]. x_start = 0, x_end = 3.
        let constraint = PeriodicityConstraint { x_start: 0.0, x_end: 3.0 };
        let order = 4; // Cubic
        let num_coeffs = 5; 
        // Knots: N+m = 5+4 = 9. Example: Clamped [0,3]
        // t0-t3=0, t4=1, t5=2, t6-t9=3
        let knots = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]); 
        // If x_start=0, only B_0,4(0) will be 1, others 0.
        // If x_end=3, only B_N-1,4(3) = B_4,4(3) will be 1, others 0.
        // So, for j=0: B_0,4(0) - B_0,4(3) = 1.0 - 0.0 = 1.0
        // For j=1: B_1,4(0) - B_1,4(3) = 0.0 - 0.0 = 0.0
        // ...
        // For j=4: B_4,4(0) - B_4,4(3) = 0.0 - 1.0 = -1.0
        // Expected a_row: [1.0, 0.0, 0.0, 0.0, -1.0]

        // Need to adjust knots for the example to be fully clamped at x_end=3 for B_N-1 to be 1.
        // Knots: t0-t3=0, t4=1, t5=2, t6-t9=3. Incorrect.
        // Knots: t0..t_{m-1} = x_min, t_N..t_{N+m-1} = x_max
        // N=5, m=4. x_min=0, x_max=3.
        // t0,t1,t2,t3 = 0
        // t_N = t5. So t_N..t_{N+m-1} = t5,t6,t7,t8.
        // For clamped [0,3], it should be:
        // knots[0..3] = 0, knots[5..8] = 3. knots[4] is the interior knot.
        // So, [0,0,0,0, 1.5, 3,3,3,3] (example with one interior knot at 1.5)
        let knots_clamped = arr1(&[0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 3.0, 3.0, 3.0]);

        let row = generate_periodicity_lp_rows(&constraint, order, &knots_clamped, num_coeffs).unwrap();
        
        assert_array_eq_tol(&row.a_row, &arr1(&[1.0, 0.0, 0.0, 0.0, -1.0]), TOL_POINTWISE);
        assert!((row.rhs - 0.0).abs() < TOL_POINTWISE);
        assert_eq!(row.kind, PointwiseConstraintKind::Equals);
    }

    #[test]
    fn test_generate_periodicity_lp_rows_validation_order_zero() {
        let constraint = PeriodicityConstraint { x_start: 0.0, x_end: 1.0 };
        let knots = arr1(&[0.0, 0.0, 1.0, 1.0]);
        let result = generate_periodicity_lp_rows(&constraint, 0, &knots, 2);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Order must be greater than 0.");
    }

    #[test]
    fn test_generate_periodicity_lp_rows_validation_num_coeffs_zero() {
        let constraint = PeriodicityConstraint { x_start: 0.0, x_end: 1.0 };
        let knots = arr1(&[0.0, 0.0]); // For N=0, m=2 -> len 2
        let result = generate_periodicity_lp_rows(&constraint, 2, &knots, 0);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Number of coefficients must be greater than 0.");
    }

    #[test]
    fn test_generate_periodicity_lp_rows_validation_knots_len() {
        let constraint = PeriodicityConstraint { x_start: 0.0, x_end: 1.0 };
        let order = 2;
        let num_coeffs = 2;
        let knots_wrong_len = arr1(&[0.0, 0.0, 1.0]); // Expected N+m = 2+2=4
        let result = generate_periodicity_lp_rows(&constraint, order, &knots_wrong_len, num_coeffs);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "Knots length must be equal to num_coefficients + order (3 != 2 + 2)."
        );
    }
}
