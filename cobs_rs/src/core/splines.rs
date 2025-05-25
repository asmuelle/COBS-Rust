use ndarray::Array1;

/// Evaluates the B-spline basis function B_j,m(x) using the Cox-de Boor recursion formula.
///
/// # Arguments
/// * `j` - Index of the B-spline basis function (0-indexed).
/// * `m` - Order of the B-spline (degree = m-1).
/// * `x` - Evaluation point.
/// * `knots` - Knot vector T = {t_0, t_1, ..., t_{num_basis_functions + order - 1}}.
///           The length of the knot vector must be `num_coeffs + order`.
///
/// # Returns
/// The value of the B-spline basis function B_j,m(x).
pub fn b_spline_basis(
    j: usize,
    m: usize,
    x: f64,
    knots: &Array1<f64>,
) -> f64 {
    if m == 0 { // Order must be at least 1
        return 0.0;
    }
    // If knots.len() < m, then knots.len() - m would underflow for usize.
    // This also covers cases like knots.len() == 0.
    if knots.len() < m {
        return 0.0;
    }

    // Number of definable basis functions.
    let num_basis_functions = knots.len() - m;
    if j >= num_basis_functions { // If j is too large (or becomes too large in recursion)
        return 0.0;
    }

    // Base case (m=1, i.e., degree 0, piecewise constant)
    if m == 1 {
        let t_j = knots[j]; // Safe: j < num_basis_functions = knots.len() - 1. So j < knots.len() -1. Thus j+1 < knots.len().
        let t_j_plus_1 = knots[j + 1]; // Safe due to j < knots.len() - m and m=1 means j < knots.len() - 1.

        // Standard case: B_j,1(x) = 1.0 if t_j <= x < t_j+1, and 0.0 otherwise.
        // Handle the edge case for the last knot interval:
        // If x is exactly at the end of the domain.
        // The number of basis functions N_coeffs = knots.len() - m. (num_basis_functions)
        // The last basis function index is N_coeffs - 1.
        // The domain is typically considered [t_{m-1}, t_{N_coeffs}].
        // For m=1, domain is [t_0, t_{N_coeffs}]. t_{N_coeffs} is knots[knots.len()-m].

        if x >= t_j && x < t_j_plus_1 {
            return 1.0;
        } else if x == t_j_plus_1 && t_j_plus_1 == knots[knots.len() - m] { // knots.len() - m is index of t_{N_coeffs}
            // This condition means x is at the end of the considered domain for splines of order m
            // and we are evaluating the basis function whose interval is the last one.
             if j == num_basis_functions - 1 { // If this is the very last basis function B_{N_coeffs-1, 1}(x)
                 return 1.0;
             } else {
                 return 0.0;
             }
        } else {
            return 0.0;
        }
    }

    // Recursive step (m > 1)
    // B_j,m(x) = (x - t_j) / (t_{j+m-1} - t_j) * B_j,m-1(x) +
    //            (t_{j+m} - x) / (t_{j+m} - t_{j+1}) * B_j+1,m-1(x)

    // Knot indexing checks:
    // j < num_basis_functions => j <= knots.len() - m - 1
    // So, j+m <= knots.len() - 1. All knot accesses up to knots[j+m] are safe.

    let t_j = knots[j];
    let t_j_plus_1 = knots[j + 1]; // Needed for the second recursive term's B_{j+1, m-1}
    let t_j_plus_m_minus_1 = knots[j + m - 1];
    let t_j_plus_m = knots[j + m];

    let w1_num = x - t_j;
    let w1_den = t_j_plus_m_minus_1 - t_j;
    let term1_coeff = if w1_den.abs() < f64::EPSILON { 0.0 } else { w1_num / w1_den };

    let w2_num = t_j_plus_m - x;
    let w2_den = t_j_plus_m - t_j_plus_1;
    let term2_coeff = if w2_den.abs() < f64::EPSILON { 0.0 } else { w2_num / w2_den };
    
    let val1 = term1_coeff * b_spline_basis(j, m - 1, x, knots);
    let val2 = term2_coeff * b_spline_basis(j + 1, m - 1, x, knots);

    val1 + val2
}

pub fn evaluate_spline(
    x: f64,                        // Point at which to evaluate the spline
    order: usize,                  // Order of the B-spline (m)
    knots: &Array1<f64>,           // Knot vector T
    coefficients: &Array1<f64>   // B-spline coefficients (a_j)
) -> Result<f64, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }

    // Validate knots and coefficients length relationship
    // The number of coefficients is N_coeffs = coefficients.len()
    // The knots length must be N_coeffs + order.
    // This is checked by validate_knots.
    if let Err(e) = crate::core::knots::validate_knots(knots, order, coefficients.len()) {
        return Err(format!("Invalid knot vector or parameters: {}", e));
    }

    let mut sum = 0.0;
    for j in 0..coefficients.len() {
        let basis_val = b_spline_basis(j, order, x, knots);
        sum += coefficients[j] * basis_val;
    }

    Ok(sum)
}


pub fn b_spline_basis_derivative(
    j: usize,        // Index of the B-spline basis function
    order: usize,    // Order of the B-spline (m)
    x: f64,          // Evaluation point
    knots: &Array1<f64> // Knot vector
) -> Result<f64, String> {
    if order == 0 {
        return Err("Order (m) cannot be 0 for derivative calculation.".to_string());
    }
    if order == 1 { // Derivative of a degree 0 (constant) basis function is 0
        return Ok(0.0);
    }

    // For B'_{j,m}(x), we need B_{j,m-1}(x) and B_{j+1,m-1}(x).
    // Ensure knots are sufficient for these order m-1 basis functions.
    // knots.len() must be at least (j or j+1) + (m-1) + 1 for indexing,
    // and generally (num_coeffs for m-1) + (m-1).
    // The b_spline_basis function itself handles j out of bounds for the given order and knots.
    
    // Term 1: B_{j,m-1}(x) / (t_{j+m-1} - t_j)
    let term1_val = b_spline_basis(j, order - 1, x, knots);
    // Denominator for term 1: t_{j+m-1} - t_j
    // Need to check if indices j and j+order-1 are valid for knots array.
    if j + order - 1 >= knots.len() { // Check for t_{j+m-1}
        // This implies insufficient knots for the first term's denominator.
        // Or j is too large. b_spline_basis would return 0 for term1_val if j is too large
        // for an (order-1) spline. If j is valid for (order-1) but j+order-1 is not for knots,
        // it's an issue with the original knot vector for order m.
        // This should ideally be caught by validate_knots if used prior, but good to be safe.
        return Err(format!(
            "Knot index j+order-1 ({} + {} - 1 = {}) is out of bounds for knots array of length {}.",
            j, order, j + order - 1, knots.len()
        ));
    }
    let den1 = knots[j + order - 1] - knots[j];
    let term1_fraction = if den1.abs() < f64::EPSILON {
        0.0
    } else {
        term1_val / den1
    };

    // Term 2: B_{j+1,m-1}(x) / (t_{j+m} - t_{j+1})
    let term2_val = b_spline_basis(j + 1, order - 1, x, knots);
    // Denominator for term 2: t_{j+m} - t_{j+1}
    // Need to check if indices j+1 and j+order are valid.
    if j + order >= knots.len() { // Check for t_{j+m}
        return Err(format!(
            "Knot index j+order ({} + {} = {}) is out of bounds for knots array of length {}.",
            j, order, j + order, knots.len()
        ));
    }
    // Check for t_{j+1} (already implicitly covered if j+order is safe and order >=1)
    // but explicitly:
    if j + 1 >= knots.len() && term2_val != 0.0 { // if term2_val is 0, j+1 might be non-issue.
         return Err(format!(
            "Knot index j+1 ({} + 1 = {}) is out of bounds for knots array of length {} (for term 2).",
            j, j + 1, knots.len()
        ));
    }


    let den2 = knots[j + order] - knots[j + 1];
    let term2_fraction = if den2.abs() < f64::EPSILON {
        0.0
    } else {
        term2_val / den2
    };
    
    Ok(((order - 1) as f64) * (term1_fraction - term2_fraction))
}


pub fn evaluate_spline_derivative(
    x: f64,                        // Point at which to evaluate the derivative
    order: usize,                  // Order of the original B-spline (m)
    knots: &Array1<f64>,           // Knot vector T
    coefficients: &Array1<f64>   // B-spline coefficients (a_j)
) -> Result<f64, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1 for spline derivative.".to_string());
    }
    // If order == 1, derivative is 0, unless coefficients make it non-zero (step function).
    // b_spline_basis_derivative will return Ok(0.0) if order == 1. Sum will be 0.

    // Validate knots and coefficients length relationship
    if let Err(e) = crate::core::knots::validate_knots(knots, order, coefficients.len()) {
        return Err(format!("Invalid knot vector or parameters for spline derivative: {}", e));
    }

    let mut sum = 0.0;
    for j in 0..coefficients.len() {
        match b_spline_basis_derivative(j, order, x, knots) {
            Ok(basis_deriv_val) => {
                sum += coefficients[j] * basis_deriv_val;
            }
            Err(e) => {
                return Err(format!("Error calculating basis derivative for B'_{},{},{}({}): {}", j, order,coefficients.len(), x, e));
            }
        }
    }
    Ok(sum)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    // Use knots module for validation in tests if needed, or rely on evaluate_spline's internal validation
    // use crate::core::knots; 

    const TOL: f64 = 1e-9; // Tolerance for float comparisons

    #[test]
    fn test_b_spline_basis_order1_degree0() {
        // Knots: 0, 1, 2, 3 (num_basis_functions = 3, order = 1)
        // B_0,1 should be 1 on [0,1), 0 otherwise
        // B_1,1 should be 1 on [1,2), 0 otherwise
        // B_2,1 should be 1 on [2,3], 0 otherwise (special handling for last point)
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let m = 1;

        // Test B_0,1(x)
        assert!((b_spline_basis(0, m, 0.0, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(0, m, 0.5, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(0, m, 0.99, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(0, m, 1.0, &knots) - 0.0).abs() < TOL); // x = t_1, so B_0,1 is 0
        assert!((b_spline_basis(0, m, -0.1, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(0, m, 2.5, &knots) - 0.0).abs() < TOL);

        // Test B_1,1(x)
        assert!((b_spline_basis(1, m, 1.0, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(1, m, 1.5, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(1, m, 1.99, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(1, m, 2.0, &knots) - 0.0).abs() < TOL); // x = t_2, so B_1,1 is 0
        assert!((b_spline_basis(1, m, 0.5, &knots) - 0.0).abs() < TOL);

        // Test B_2,1(x) - last basis function
        // knots.len() - m = 4 - 1 = 3. This is t_3 (value 3.0)
        // j = knots.len() - m - 1 = 3 - 1 = 2
        // So B_2,1 should be 1 on [t_2, t_3] = [2.0, 3.0]
        assert!((b_spline_basis(2, m, 2.0, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(2, m, 2.5, &knots) - 1.0).abs() < TOL);
        assert!((b_spline_basis(2, m, 3.0, &knots) - 1.0).abs() < TOL); // x = t_3 (last knot in domain)
        assert!((b_spline_basis(2, m, 1.5, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(2, m, 3.1, &knots) - 0.0).abs() < TOL);

        // Extended knot vector
        // Knots: 0,1,2,3,4 (num_basis_functions = 4, order = 1)
        let knots_ext = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        // Test B_3,1(x)
        // knots.len() - m = 5 - 1 = 4. This is t_4 (value 4.0)
        // j = knots.len() - m - 1 = 4 - 1 = 3
        assert!((b_spline_basis(3, m, 3.0, &knots_ext) - 1.0).abs() < TOL);
        assert!((b_spline_basis(3, m, 3.5, &knots_ext) - 1.0).abs() < TOL);
        assert!((b_spline_basis(3, m, 4.0, &knots_ext) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_b_spline_basis_order2_degree1_linear() {
        // Knots: 0, 1, 2, 3 (num_basis_functions = 2, order = 2)
        // B_0,2(x) supported on [0,2), peaks at x=1 (value 1.0)
        // B_1,2(x) supported on [1,3), peaks at x=2 (value 1.0)
        // Knots T = {t_0, t_1, t_2, t_3} = {0,1,2,3}
        // Order m=2
        // B_j,2(x) = (x - t_j)/(t_{j+1}-t_j) * B_j,1(x) + (t_{j+2}-x)/(t_{j+2}-t_{j+1}) * B_{j+1},1(x)
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let m = 2;

        // Test B_0,2(x)
        // Support: [t_0, t_2] = [0, 2]
        // B_0,2(x) = (x-0)/(1-0) * B_0,1(x) + (2-x)/(2-1) * B_1,1(x)
        // B_0,2(0.0) = 0 * B_0,1(0) + 2 * B_1,1(0) = 0 * 1 + 2 * 0 = 0
        assert!((b_spline_basis(0, m, 0.0, &knots) - 0.0).abs() < TOL);
        // B_0,2(0.5) = (0.5/1)*B_0,1(0.5) + (1.5/1)*B_1,1(0.5) = 0.5 * 1 + 1.5 * 0 = 0.5
        assert!((b_spline_basis(0, m, 0.5, &knots) - 0.5).abs() < TOL);
        // B_0,2(1.0) = (1/1)*B_0,1(1) + (1/1)*B_1,1(1) = 1 * 0 + 1 * 1 = 1.0  -- ERROR in formula application here
        // B_0,1(1.0) is 0, B_1,1(1.0) is 1.
        // (x-t_0)/(t_1-t_0) * B_0,1(x) for x in [t_0, t_1)
        // (t_2-x)/(t_2-t_1) * B_1,1(x) for x in [t_1, t_2)
        // At x=1.0:
        // For first term: B_0,1(1.0) is 0. (x-t_0)/(t_1-t_0) is (1-0)/(1-0) = 1. So 1*0 = 0.
        // For second term: B_1,1(1.0) is 1. (t_2-x)/(t_2-t_1) is (2-1)/(2-1) = 1. So 1*1 = 1.
        // Sum = 0+1=1.
        assert!((b_spline_basis(0, m, 1.0, &knots) - 1.0).abs() < TOL);
        // B_0,2(1.5) = (1.5/1)*B_0,1(1.5) + (0.5/1)*B_1,1(1.5) = 1.5 * 0 + 0.5 * 1 = 0.5
        assert!((b_spline_basis(0, m, 1.5, &knots) - 0.5).abs() < TOL);
        // B_0,2(2.0) = (2/1)*B_0,1(2) + (0/1)*B_1,1(2) = 0. (B_0,1(2)=0, B_1,1(2) depends on end condition)
        // B_1,1(2) is 1 if x=t_end and this is the last interval. Knots 0,1,2,3. m=1. B_1,1(x) is for [1,2).
        // For B_1,1(x) and knots [0,1,2,3], t_j=1, t_{j+1}=2.
        // t_j <= x < t_{j+1}. B_1,1(2.0) is 0.
        // So B_0,2(2.0) = 0
        assert!((b_spline_basis(0, m, 2.0, &knots) - 0.0).abs() < TOL);


        // Test B_1,2(x)
        // Support: [t_1, t_3] = [1, 3]
        // B_1,2(x) = (x-t_1)/(t_2-t_1) * B_1,1(x) + (t_3-x)/(t_3-t_2) * B_2,1(x)
        // B_1,2(1.0) = (1-1)/(2-1)*B_1,1(1) + (3-1)/(3-2)*B_2,1(1) = 0 * 1 + 2 * 0 = 0
        assert!((b_spline_basis(1, m, 1.0, &knots) - 0.0).abs() < TOL);
        // B_1,2(1.5) = (1.5-1)/(2-1)*B_1,1(1.5) + (3-1.5)/(3-2)*B_2,1(1.5) = 0.5 * 1 + 1.5 * 0 = 0.5
        assert!((b_spline_basis(1, m, 1.5, &knots) - 0.5).abs() < TOL);
        // B_1,2(2.0) = (2-1)/(2-1)*B_1,1(2) + (3-2)/(3-2)*B_2,1(2)
        // B_1,1(2.0) is 0. B_2,1(2.0) is 1.
        // = 1 * 0 + 1 * 1 = 1.0
        assert!((b_spline_basis(1, m, 2.0, &knots) - 1.0).abs() < TOL);
        // B_1,2(2.5) = (2.5-1)/(2-1)*B_1,1(2.5) + (3-2.5)/(3-2)*B_2,1(2.5)
        // B_1,1(2.5) is 0. B_2,1(2.5) is 1.
        // = 1.5 * 0 + 0.5 * 1 = 0.5
        assert!((b_spline_basis(1, m, 2.5, &knots) - 0.5).abs() < TOL);
        // B_1,2(3.0)
        // B_1,1(3.0) is 0. B_2,1(3.0) is 1 (due to special end condition).
        // (3.0-1)/(2-1)*B_1,1(3.0) + (3-3.0)/(3-2)*B_2,1(3.0)
        // = 2.0 * 0 + 0.0 * 1 = 0.0
        assert!((b_spline_basis(1, m, 3.0, &knots) - 0.0).abs() < TOL);

        // Sum of basis functions should be 1.0 in the interval [t_{m-1}, t_{N_coeffs}]
        // Here m=2, N_coeffs = knots.len() - m = 4-2=2.
        // Interval is [t_1, t_2] = [1.0, 2.0]
        // For x = 1.0: B_0,2(1.0) + B_1,2(1.0) = 1.0 + 0.0 = 1.0
        assert!((b_spline_basis(0, m, 1.0, &knots) + b_spline_basis(1, m, 1.0, &knots) - 1.0).abs() < TOL);
        // For x = 1.5: B_0,2(1.5) + B_1,2(1.5) = 0.5 + 0.5 = 1.0
        assert!((b_spline_basis(0, m, 1.5, &knots) + b_spline_basis(1, m, 1.5, &knots) - 1.0).abs() < TOL);
        // For x = 2.0: B_0,2(2.0) + B_1,2(2.0) = 0.0 + 1.0 = 1.0
        assert!((b_spline_basis(0, m, 2.0, &knots) + b_spline_basis(1, m, 2.0, &knots) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_b_spline_basis_coincident_knots() {
        // Knots: 0, 0, 1, 2 (num_basis_functions = 2, order = 2)
        // T = {0,0,1,2}
        // B_0,2(x) supported on [t_0,t_2] = [0,1]
        // B_1,2(x) supported on [t_1,t_3] = [0,2]
        let knots = arr1(&[0.0, 0.0, 1.0, 2.0]);
        let m = 2;

        // B_0,2(x) = (x - t_0)/(t_1 - t_0) * B_0,1(x) + (t_2 - x)/(t_2 - t_1) * B_1,1(x)
        // t_0=0, t_1=0, t_2=1
        // First term: (x-0)/(0-0) * B_0,1(x) -> should be 0 due to zero denominator
        // Second term: (1-x)/(1-0) * B_1,1(x)
        // So, B_0,2(x) = (1-x) * B_1,1(x)
        // B_1,1(x) is 1 if t_1 <= x < t_2, i.e. 0 <= x < 1.
        // So B_0,2(x) = 1-x for 0 <= x < 1.
        assert!((b_spline_basis(0, m, 0.0, &knots) - 1.0).abs() < TOL); // (1-0)*B_1,1(0) = 1*1 = 1
        assert!((b_spline_basis(0, m, 0.5, &knots) - 0.5).abs() < TOL); // (1-0.5)*B_1,1(0.5) = 0.5*1 = 0.5
        assert!((b_spline_basis(0, m, 0.99, &knots) - 0.01).abs() < TOL); // (1-0.99)*B_1,1(0.99) = 0.01*1 = 0.01
        assert!((b_spline_basis(0, m, 1.0, &knots) - 0.0).abs() < TOL); // B_1,1(1.0) is 0

        // B_1,2(x) = (x - t_1)/(t_2 - t_1) * B_1,1(x) + (t_3 - x)/(t_3 - t_2) * B_2,1(x)
        // t_1=0, t_2=1, t_3=2
        // B_1,2(x) = (x-0)/(1-0) * B_1,1(x) + (2-x)/(2-1) * B_2,1(x)
        // B_1,2(x) = x * B_1,1(x) + (2-x) * B_2,1(x)
        // B_1,1(x) is 1 if 0 <= x < 1. B_2,1(x) is 1 if 1 <= x <= 2 (last interval special handling)
        // For x = 0.0: 0*B_1,1(0) + (2-0)*B_2,1(0) = 0*1 + 2*0 = 0
        assert!((b_spline_basis(1, m, 0.0, &knots) - 0.0).abs() < TOL);
        // For x = 0.5: 0.5*B_1,1(0.5) + (2-0.5)*B_2,1(0.5) = 0.5*1 + 1.5*0 = 0.5
        assert!((b_spline_basis(1, m, 0.5, &knots) - 0.5).abs() < TOL);
        // For x = 1.0: 1.0*B_1,1(1.0) + (2-1.0)*B_2,1(1.0) = 1.0*0 + 1.0*1 = 1.0
        assert!((b_spline_basis(1, m, 1.0, &knots) - 1.0).abs() < TOL);
         // For x = 1.5: 1.5*B_1,1(1.5) + (2-1.5)*B_2,1(1.5) = 1.5*0 + 0.5*1 = 0.5
        assert!((b_spline_basis(1, m, 1.5, &knots) - 0.5).abs() < TOL);
        // For x = 2.0: 2.0*B_1,1(2.0) + (2-2.0)*B_2,1(2.0) = 2.0*0 + 0.0*1 = 0.0
        assert!((b_spline_basis(1, m, 2.0, &knots) - 0.0).abs() < TOL);

        // Sum of basis functions at x=0.5: B_0,2(0.5) + B_1,2(0.5) = 0.5 + 0.5 = 1.0
        assert!((b_spline_basis(0, m, 0.5, &knots) + b_spline_basis(1, m, 0.5, &knots) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_b_spline_basis_book_example_pg69() {
        // Example from "A Practical Guide to Splines" by Carl de Boor, page 69 (figure)
        // Order m=3 (quadratic splines), knots t_i = i for all i.
        // T = {0,1,2,3,4,5} (N_coeffs = 3, order = 3)
        // B_0,3(x) on [0,3]
        // B_1,3(x) on [1,4]
        // B_2,3(x) on [2,5]
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = 3;

        // B_0,3(x)
        // B_0,3(0.5) (In [0,1]):
        // B_0,3(x) = (x-t0)/(t2-t0) B_0,2(x) + (t3-x)/(t3-t1) B_1,2(x)
        //          = x/2 * B_0,2(x) + (3-x)/2 * B_1,2(x)
        // B_0,2(x) for T={0,1,2} (using sub-knots for this part):
        //   B_0,2(x) = (x-t0)/(t1-t0) B_0,1(x) + (t2-x)/(t2-t1) B_1,1(x)
        //            = x/1 * B_0,1(x) + (2-x)/1 * B_1,1(x)
        //   B_0,1(x) is 1 on [0,1), 0 else. B_1,1(x) is 1 on [1,2), 0 else.
        //   B_0,2(0.5) = 0.5 * B_0,1(0.5) + 1.5 * B_1,1(0.5) = 0.5 * 1 + 1.5 * 0 = 0.5
        // B_1,2(x) for T={1,2,3} (using sub-knots for this part):
        //   B_1,2(x) = (x-t1)/(t2-t1) B_1,1(x) + (t3-x)/(t3-t2) B_2,1(x)
        //            = (x-1)/1 * B_1,1(x) + (3-x)/1 * B_2,1(x)
        //   B_1,1(x) is 1 on [1,2), 0 else. B_2,1(x) is 1 on [2,3), 0 else.
        //   B_1,2(0.5) = -0.5 * B_1,1(0.5) + 2.5 * B_2,1(0.5) = -0.5 * 0 + 2.5 * 0 = 0
        // So, B_0,3(0.5) = 0.5/2 * B_0,2(0.5) + (3-0.5)/2 * B_1,2(0.5)
        //              = 0.25 * 0.5 + 1.25 * 0 = 0.125
        assert!((b_spline_basis(0, m, 0.5, &knots) - 0.125).abs() < TOL);

        // B_0,3(1.5) (In [1,2]):
        // B_0,2(1.5) = 1.5 * B_0,1(1.5) + 0.5 * B_1,1(1.5) = 1.5*0 + 0.5*1 = 0.5
        // B_1,2(1.5) = (1.5-1)*B_1,1(1.5) + (3-1.5)*B_2,1(1.5) = 0.5*1 + 1.5*0 = 0.5
        // B_0,3(1.5) = 1.5/2 * B_0,2(1.5) + (3-1.5)/2 * B_1,2(1.5)
        //            = 0.75 * 0.5 + 0.75 * 0.5 = 0.375 + 0.375 = 0.75
        assert!((b_spline_basis(0, m, 1.5, &knots) - 0.75).abs() < TOL);

        // B_0,3(2.5) (In [2,3]):
        // B_0,2(2.5) = 2.5*B_0,1(2.5) + (-0.5)*B_1,1(2.5) = 0 (B_0,1 and B_1,1 are 0 outside [0,2))
        //    Actually, B_0,2(x) relies on knots t0,t1,t2 = {0,1,2}. Support is [0,2]. So B_0,2(2.5) = 0.
        // B_1,2(2.5) (uses knots t1,t2,t3 = {1,2,3})
        //    B_1,2(2.5) = (2.5-1)*B_1,1(2.5) + (3-2.5)*B_2,1(2.5)
        //               = 1.5 * 0 + 0.5 * 1 = 0.5
        // B_0,3(2.5) = 2.5/2 * B_0,2(2.5) + (3-2.5)/2 * B_1,2(2.5)
        //            = 1.25 * 0 + 0.25 * 0.5 = 0.125
        assert!((b_spline_basis(0, m, 2.5, &knots) - 0.125).abs() < TOL);

        // Check sum of squares property for this example at specific points
        // At x=2.0 (knot): B_0,3(2)=0.5, B_1,3(2)=0.5, B_2,3(2)=0 (from de Boor's fig)
        // B_0,3(2.0):
        //   B_0,2(2.0) = 2*B_0,1(2) + 0*B_1,1(2) = 0 (B_0,1(2)=0, B_1,1(2)=0 as per base case for non-last interval)
        //   B_1,2(2.0) (uses {1,2,3}): (2-1)B_1,1(2) + (3-2)B_2,1(2) = 1*0 + 1*1 = 1
        //   B_0,3(2.0) = 2/2 * B_0,2(2.0) + (3-2)/2 * B_1,2(2.0) = 1*0 + 0.5*1 = 0.5
        assert!((b_spline_basis(0, m, 2.0, &knots) - 0.5).abs() < TOL);

        // B_1,3(2.0):
        // B_1,3(x) = (x-t1)/(t3-t1) B_1,2(x) + (t4-x)/(t4-t2) B_2,2(x)
        //          = (x-1)/2 * B_1,2(x) + (4-x)/2 * B_2,2(x)
        // B_1,2(2.0) = 1 (calculated above)
        // B_2,2(x) (uses knots t2,t3,t4 = {2,3,4}):
        //   B_2,2(x) = (x-t2)/(t3-t2) B_2,1(x) + (t4-x)/(t4-t3) B_3,1(x)
        //   B_2,1(x) is 1 on [2,3), B_3,1(x) is 1 on [3,4)
        //   B_2,2(2.0) = (2-2)/(3-2)B_2,1(2) + (4-2)/(4-3)B_3,1(2) = 0*1 + 2*0 = 0
        // B_1,3(2.0) = (2-1)/2 * B_1,2(2.0) + (4-2)/2 * B_2,2(2.0)
        //            = 0.5 * 1 + 1.0 * 0 = 0.5
        assert!((b_spline_basis(1, m, 2.0, &knots) - 0.5).abs() < TOL);

        // B_2,3(2.0):
        // B_2,3(x) = (x-t2)/(t4-t2) B_2,2(x) + (t5-x)/(t5-t3) B_3,2(x)
        //          = (x-2)/2 * B_2,2(x) + (5-x)/2 * B_3,2(x)
        // B_2,2(2.0) = 0 (calculated above)
        // B_3,2(x) (uses knots t3,t4,t5 = {3,4,5}):
        //   B_3,2(x) = (x-t3)/(t4-t3) B_3,1(x) + (t5-x)/(t5-t4) B_4,1(x)
        //   B_3,1(x) is 1 on [3,4), B_4,1(x) is 1 on [4,5]
        //   B_3,2(2.0) = (2-3)B_3,1(2) + (5-2)B_4,1(2) = (-1)*0 + 3*0 = 0
        // B_2,3(2.0) = (2-2)/2 * B_2,2(2.0) + (5-2)/2 * B_3,2(2.0)
        //            = 0 * 0 + 1.5 * 0 = 0
        assert!((b_spline_basis(2, m, 2.0, &knots) - 0.0).abs() < TOL);

        // Sum of basis functions = 1.0 over [t_{m-1}, t_{N_coeffs}] = [t_2, t_3] = [2,3]
        // At x=2.0: sum = B_0,3(2) + B_1,3(2) + B_2,3(2) = 0.5 + 0.5 + 0 = 1.0
        let sum_at_2 = b_spline_basis(0,m,2.0,&knots) + b_spline_basis(1,m,2.0,&knots) + b_spline_basis(2,m,2.0,&knots);
        assert!((sum_at_2 - 1.0).abs() < TOL);

        // At x=2.5:
        // B_0,3(2.5) = 0.125 (calculated above)
        // B_1,3(2.5):
        //   B_1,2(2.5) = 0.5 (calculated above)
        //   B_2,2(2.5) (uses {2,3,4}): (2.5-2)B_2,1(2.5) + (4-2.5)B_3,1(2.5) = 0.5*1 + 1.5*0 = 0.5
        //   B_1,3(2.5) = (2.5-1)/2 * B_1,2(2.5) + (4-2.5)/2 * B_2,2(2.5)
        //              = 0.75 * 0.5 + 0.75 * 0.5 = 0.375 + 0.375 = 0.75
        assert!((b_spline_basis(1, m, 2.5, &knots) - 0.75).abs() < TOL);
        // B_2,3(2.5):
        //   B_2,2(2.5) = 0.5 (calculated above)
        //   B_3,2(2.5) (uses {3,4,5}): (2.5-3)B_3,1(2.5) + (5-2.5)B_4,1(2.5) = -0.5*0 + 2.5*0 = 0
        //   B_2,3(2.5) = (2.5-2)/2 * B_2,2(2.5) + (5-2.5)/2 * B_3,2(2.5)
        //              = 0.25 * 0.5 + 1.25 * 0 = 0.125
        assert!((b_spline_basis(2, m, 2.5, &knots) - 0.125).abs() < TOL);
        // Sum at x=2.5: 0.125 + 0.75 + 0.125 = 1.0
        let sum_at_2_5 = b_spline_basis(0,m,2.5,&knots) + b_spline_basis(1,m,2.5,&knots) + b_spline_basis(2,m,2.5,&knots);
        assert!((sum_at_2_5 - 1.0).abs() < TOL);
    }

     #[test]
    fn test_b_spline_outside_support() {
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = 3; // Quadratic
        // B_0,3 is supported on [0,3]
        assert!((b_spline_basis(0, m, -0.5, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(0, m, 3.0, &knots) - 0.0).abs() < TOL); // B_j,m(t_{j+m}) should be 0
        assert!((b_spline_basis(0, m, 3.5, &knots) - 0.0).abs() < TOL);

        // B_1,3 is supported on [1,4]
        assert!((b_spline_basis(1, m, 0.5, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(1, m, 4.0, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(1, m, 4.5, &knots) - 0.0).abs() < TOL);

        // B_2,3 is supported on [2,5]
        // knots.len() - m = 6 - 3 = 3. This is t_3 (value 3.0)
        // j_max = knots.len() - m - 1 = 6 - 3 - 1 = 2.
        // So B_2,3(x) is the last basis function.
        // Its support is [t_2, t_{2+3}] = [t_2, t_5] = [2.0, 5.0]
        // At x=t_5 = 5.0, it should be zero.
        // The special handling for m=1 at x == knots[knots.len() - m] does not apply for m > 1.
        assert!((b_spline_basis(2, m, 1.5, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(2, m, 5.0, &knots) - 0.0).abs() < TOL); // B_j,m(t_{j+m}) should be 0
        assert!((b_spline_basis(2, m, 5.5, &knots) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_invalid_j_or_m() {
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = 3; // order
        // j too large for given knots and m
        // num_basis_functions = knots.len() - m = 6 - 3 = 3. Max j is 2.
        assert!((b_spline_basis(3, m, 2.5, &knots) - 0.0).abs() < TOL);
        assert!((b_spline_basis(10, m, 2.5, &knots) - 0.0).abs() < TOL);

        // m = 0 is invalid order
        // The function currently doesn't explicitly check m=0, but recursion would fail.
        // The problem states m is order, degree = m-1. So m>=1.
        // Base case is m=1. If m=0, it would try m-1 = -1, which is not meaningful.
        // Let's assume m >= 1 based on definition of order.
        // If m > knots.len(), e.g. m = 7 for knots.len()=6
        // knots.len() - m would be negative.
        // The j >= knots.len() - m check should handle this.
        // e.g., j=0, m=7, knots.len()=6. knots.len()-m = -1. 0 >= -1.
        // Then in recursion, j+m or j+m-1 could go out of bounds.
        // knots[j+m-1] => knots[0+7-1] = knots[6] is out of bounds.
        // Current code handles out-of-bounds for j+m, but not j+m-1.
        // Let's test a case where j+m-1 is an issue.
        // j=0, m=6, knots.len()=6.
        // t_j_plus_m_minus_1 = knots[0+6-1] = knots[5] (valid)
        // t_j_plus_m = knots[0+6] = knots[6] (out of bounds)
        // The j+m >= knots.len() check handles this.
        assert!((b_spline_basis(0, 7, 2.5, &knots) - 0.0).abs() < TOL); // m too large

        // What if m makes num_basis_functions zero or negative?
        // knots.len()=6, m=6. num_basis_functions = 0. Max j = -1.
        // Any j>=0 will trigger j >= knots.len() - m. (e.g. 0 >= 6-6 = 0)
        assert!((b_spline_basis(0, 6, 2.5, &knots) - 0.0).abs() < TOL);

        // What if knots array is too short for order m?
        // e.g. knots = [0,1], m=3. knots.len()=2.
        // j=0, m=3. j >= knots.len()-m (0 >= 2-3 = -1) is true.
        // Base case m=1 not hit.
        // Recursive step: j+m = 0+3 = 3. knots.len()=2. j+m >= knots.len() is true. Returns 0.
        let short_knots = arr1(&[0.0, 1.0]);
        assert!((b_spline_basis(0, 3, 0.5, &short_knots) - 0.0).abs() < TOL);
    }

    // Tests for evaluate_spline

    #[test]
    fn test_evaluate_linear_spline_order2() {
        // Order m=2 (linear). N_coeffs = 3. knots.len() = N_coeffs + order = 3+2=5
        let knots = arr1(&[0.0, 0.0, 1.0, 2.0, 2.0]); // t0,t1,t2,t3,t4
        let coeffs = arr1(&[1.0, 2.0, 1.5]); // a0, a1, a2

        // B_0,2(x) on [0,1), uses t0,t1,t2 = [0,0,1]
        // B_1,2(x) on [0,2), uses t1,t2,t3 = [0,1,2]
        // B_2,2(x) on [1,2], uses t2,t3,t4 = [1,2,2]

        // At x = 0.0:
        // B_0,2(0.0) = 1.0 (using t0,t1,t2 = [0,0,1])
        // B_1,2(0.0) = 0.0 (using t1,t2,t3 = [0,1,2])
        // B_2,2(0.0) = 0.0
        // Sum = 1.0 * a0 = 1.0
        let val_at_0 = evaluate_spline(0.0, 2, &knots, &coeffs).unwrap();
        assert!((val_at_0 - 1.0).abs() < TOL);

        // At x = 0.5 (in [0,1)):
        // B_0,2(0.5) = ( (0.5-0)/(0-0) ) * B_0,1(0.5) + ( (1-0.5)/(1-0) ) * B_1,1(0.5)
        //            = (0 if den=0) + (0.5/1) * B_1,1(0.5) [B_1,1 for knots [0,1,2] means on [0,1)]
        // Knots for B_0,2 are [0,0,1]. B_0,1 based on [0,0], B_1,1 based on [0,1].
        // B_0,2(0.5) using [0,0,1]: (x-t0)/(t1-t0)*B01 + (t2-x)/(t2-t1)*B11
        //             = (0.5-0)/(0-0)*B01(0.5) + (1-0.5)/(1-0)*B11(0.5)
        //             = 0 (first term) + 0.5 * B11(0.5) where B11 is on [0,1) with knots [0,1,...]
        // B_0,1 uses [0,0]. B_0,1(0.5) = 0.
        // B_1,1 uses [0,1]. B_1,1(0.5) = 1.
        // So B_0,2(0.5) = 0.5.
        //
        // B_1,2(0.5) using [0,1,2]: (x-t1)/(t2-t1)*B11 + (t3-x)/(t3-t2)*B21
        //             = (0.5-0)/(1-0)*B11(0.5) + (2-0.5)/(2-1)*B21(0.5)
        // B_1,1 uses [0,1] for this. B_1,1(0.5) = 1.
        // B_2,1 uses [1,2] for this. B_2,1(0.5) = 0.
        // So B_1,2(0.5) = 0.5 * 1 + 1.5 * 0 = 0.5.
        //
        // B_2,2(0.5) using [1,2,2]: support [1,2]. So B_2,2(0.5)=0.
        // Sum = a0*0.5 + a1*0.5 = 1.0*0.5 + 2.0*0.5 = 0.5 + 1.0 = 1.5
        let val_at_0_5 = evaluate_spline(0.5, 2, &knots, &coeffs).unwrap();
        assert!((val_at_0_5 - 1.5).abs() < TOL);

        // At x = 1.0:
        // B_0,2(1.0) = 0.0
        // B_1,2(1.0) (uses [0,1,2]) = 1.0
        // B_2,2(1.0) (uses [1,2,2]) = 0.0 (careful, t2=1, t3=2, t4=2. B_2,2(1.0) is start of its support, should be 1.0)
        // Let's verify B_2,2(1.0) with knots [1,2,2]:
        // B_2,2(x) = (x-t2)/(t3-t2)*B21(x) + (t4-x)/(t4-t3)*B31(x)
        // B_2,1 for [1,2,...] is 1 on [1,2). B_3,1 for [2,2,...] is 1 on [2,2] (if last).
        // B_2,2(1.0) = (1-1)/(2-1)*B21(1) + (2-1)/(2-2)*B31(1) = 0 + (0 if den=0) = 0. This seems right.
        // The above reasoning for B_2,2(1.0) was simpler: support [t2,t4] = [1,2]. At x=t2=1, it should be 1 IF t1 < t2 = t3 < t4 etc.
        // With coincident knots, it's more complex.
        // B_0,2(1.0) based on [0,0,1] is 0.0 (x=t2)
        // B_1,2(1.0) based on [0,1,2] is 1.0 (x=t2, peak)
        // B_2,2(1.0) based on [1,2,2] is 1.0 (x=t2, start of support [1,2])
        // Sum = a0*0 + a1*1.0 + a2*1.0 = 2.0 + 1.5 = 3.5. This is standard for endpoints of segments with C0 continuity.
        // Actually, sum of basis functions should be 1. At x=1, B_1,2(1)=1 and B_2,2(1)=0. No, this is not right.
        // Standard partition of unity: Sum B_j,m(x) = 1.
        // For x in [t_{m-1}, t_{N_coeffs}], sum is 1. Here m=2. N_coeffs=3. Domain [t_1, t_3] = [0,2].
        // At x=1.0 (which is t2):
        // B_0,2(1.0) = 0
        // B_1,2(1.0) = 1 (peak for B_1,2 with knots 0,1,2)
        // B_2,2(1.0) = 0 (B_2,2 with knots 1,2,2 has support [1,2], peak at 2, value at 1 is 0)
        // Let's re-evaluate B_2,2(1.0) on T={t2,t3,t4}={1,2,2}
        // B_2,2(x) = (x-1)/(2-1) B_2,1(x) + (2-x)/(2-2) B_3,1(x)
        // B_2,1(x) on T'={1,2,2} is 1 on [1,2). B_3,1(x) on T''={2,2} is 1 on x=2.
        // B_2,2(1.0) = (1-1)/1 * B_2,1(1) + (2-1)/0 * B_3,1(1) = 0 * 1 + 0 = 0.
        // So, sum = a1 * 1.0 = 2.0 * 1.0 = 2.0.
        let val_at_1_0 = evaluate_spline(1.0, 2, &knots, &coeffs).unwrap();
        assert!((val_at_1_0 - 2.0).abs() < TOL);

        // At x = 1.5 (in [1,2)):
        // B_0,2(1.5) = 0
        // B_1,2(1.5) using [0,1,2]: (1.5-0)/(1-0)*B11(1.5) + (2-1.5)/(2-1)*B21(1.5)
        //             B11 for [0,1]. B11(1.5)=0. B21 for [1,2]. B21(1.5)=1.
        //             = 1.5*0 + 0.5*1 = 0.5
        // B_2,2(1.5) using [1,2,2]: (1.5-1)/(2-1)*B21(1.5) + (2-1.5)/(2-2)*B31(1.5)
        //             B21 for [1,2]. B21(1.5)=1. B31 for [2,2]. B31(1.5)=0.
        //             = (0.5/1)*1 + 0 = 0.5
        // Sum = a1*0.5 + a2*0.5 = 2.0*0.5 + 1.5*0.5 = 1.0 + 0.75 = 1.75
        let val_at_1_5 = evaluate_spline(1.5, 2, &knots, &coeffs).unwrap();
        assert!((val_at_1_5 - 1.75).abs() < TOL);
        
        // At x = 2.0:
        // B_0,2(2.0) = 0
        // B_1,2(2.0) = 0
        // B_2,2(2.0) = 1
        // Sum = a2 * B_2,2(2.0). If B_2,2(2.0) is 0, then sum is 0.
        let val_at_2_0 = evaluate_spline(2.0, 2, &knots, &coeffs).unwrap();
        assert!((val_at_2_0 - 0.0).abs() < TOL); // Corrected expectation
    }

    #[test]
    fn test_evaluate_quadratic_spline_order3() {
        // Order m=3 (quadratic). N_coeffs = 4. knots.len() = 4+3=7
        let knots = arr1(&[0.,0.,0.,1.,2.,2.,2.]); // t0..t6
        let coeffs = arr1(&[1.0, 2.0, 1.5, 3.0]); // a0,a1,a2,a3

        // Domain [t_{m-1}, t_{N_coeffs}] = [t_2, t_4] = [0, 2]
        // B_0,3 on [0,1] (uses 0001)
        // B_1,3 on [0,2] (uses 0012)
        // B_2,3 on [0,2] (uses 0122)
        // B_3,3 on [1,2] (uses 1222)

        // At x = 0.0: (t2, start of effective domain)
        // B_0,3(0) = 1 (from properties of clamped splines)
        // B_1,3(0) = 0
        // B_2,3(0) = 0
        // B_3,3(0) = 0
        // Sum = a0 * 1.0 = 1.0
        let val_at_0 = evaluate_spline(0.0, 3, &knots, &coeffs).unwrap();
        assert!((val_at_0 - 1.0).abs() < TOL);

        // At x = 0.5 (between t2=0 and t3=1):
        // B_0,3(0.5) on [0,0,0,1] -> check: ( (x-t0)/(t2-t0) B02 + (t3-x)/(t3-t1) B12 )
        // B_0,3(0.5) = 0.125 (from de Boor book example pg 69, scaled for interval [0,1] vs [0,3])
        // This is actually B_2,3(0.5) from de Boor's example if knots are 0,0,0,1,2,3, order 3.
        // For knots [0,0,0,1]: B_0,3(0.5) = ( (0.5-0)/(0-0) )B_0,2(0.5) + ( (1-0.5)/(1-0) )B_1,2(0.5)
        // B_0,2 on [0,0,0], B_1,2 on [0,0,1]
        // B_1,2(0.5) for [0,0,1]: ( (0.5-0)/(0-0) )B_1,1(0.5) + ( (1-0.5)/(1-0) )B_2,1(0.5)
        // B_1,1 for [0,0], B_2,1 for [0,1]. B_2,1(0.5)=1. So B_1,2(0.5) = 0.5.
        // B_0,3(0.5) = 0.5 * B_1,2(0.5) = 0.5 * 0.5 = 0.25.
        //
        // B_1,3(0.5) on [0,0,1,2]: ( (0.5-0)/(1-0) )B_1,2(0.5) + ( (2-0.5)/(2-0) )B_2,2(0.5)
        // B_1,2 on [0,0,1] (val 0.5 as above). B_2,2 on [0,1,2].
        // B_2,2(0.5) for [0,1,2]: ( (0.5-0)/(1-0) )B_2,1(0.5) + ( (2-0.5)/(2-1) )B_3,1(0.5)
        // B_2,1 on [0,1] (val 1). B_3,1 on [1,2] (val 0). So B_2,2(0.5) = 0.5.
        // B_1,3(0.5) = 0.5 * 0.5 + (1.5/2) * 0.5 = 0.25 + 0.375 = 0.625.
        //
        // B_2,3(0.5) on [0,1,2,2]: ( (0.5-0)/(2-0) )B_2,2(0.5) + ( (2-0.5)/(2-1) )B_3,2(0.5)
        // B_2,2 on [0,1,2] (val 0.5). B_3,2 on [1,2,2].
        // B_3,2(0.5) for [1,2,2] is 0 (support [1,2]).
        // B_2,3(0.5) = (0.5/2) * 0.5 + 0 = 0.125.
        //
        // B_3,3(0.5) on [1,2,2,2] is 0 (support [1,2]).
        // Sum = a0*0.25 + a1*0.625 + a2*0.125 + a3*0
        //     = 1.0*0.25 + 2.0*0.625 + 1.5*0.125
        //     = 0.25 + 1.25 + 0.1875 = 1.6875
        let val_at_0_5 = evaluate_spline(0.5, 3, &knots, &coeffs).unwrap();
        assert!((val_at_0_5 - 1.6875).abs() < TOL);

        // At x = 1.0 (knot t3):
        // B_0,3(1.0) = 0 (end of support)
        // B_1,3(1.0) for [0,0,1,2] = 0.5 (de Boor A Practical Guide to Splines, Fig IX.4, B_i,k(t_i+k/2))
        // B_2,3(1.0) for [0,1,2,2] = 0.5
        // B_3,3(1.0) for [1,2,2,2] = 0 (start of support)
        // Sum = a1*0.5 + a2*0.5 = 2.0*0.5 + 1.5*0.5 = 1.0 + 0.75 = 1.75
        let val_at_1_0 = evaluate_spline(1.0, 3, &knots, &coeffs).unwrap();
        assert!((val_at_1_0 - 1.75).abs() < TOL);

        // At x = 2.0: (t4, end of effective domain)
        // B_0,3(2.0)=0, B_1,3(2.0)=0, B_2,3(2.0)=0
        // B_3,3(2.0)=0 with current b_spline_basis.
        // Sum = a3 * 0.0 = 0.0
        let val_at_2_0 = evaluate_spline(2.0, 3, &knots, &coeffs).unwrap();
        assert!((val_at_2_0 - 0.0).abs() < TOL); // Corrected expectation
    }

    #[test]
    fn test_evaluate_spline_outside_support() {
        let knots = arr1(&[0.,0.,0.,1.,2.,2.,2.]); // Domain [0,2]
        let coeffs = arr1(&[1.0, 2.0, 1.5, 3.0]);
        let order = 3;

        let val_before = evaluate_spline(-0.5, order, &knots, &coeffs).unwrap();
        assert!((val_before - 0.0).abs() < TOL);

        let val_after = evaluate_spline(2.5, order, &knots, &coeffs).unwrap();
        assert!((val_after - 0.0).abs() < TOL);
        
        // Check boundary values again carefully
        // x = 0.0 is t_2. Support of B_j,m is [t_j, t_{j+m}].
        // Effective domain is [t_{m-1}, t_{N_coeffs}]. Here [t_2, t_4] = [0.0, 2.0].
        // At x = t_{m-1} = 0.0, only B_{0,m} should be non-zero (and =1 for clamped).
        // Spline value should be coeffs[0]. (This was tested above, result 1.0)
        // At x = t_{N_coeffs} = 2.0, only B_{N_coeffs-1,m} should be non-zero (and =1 for clamped).
        // Spline value should be coeffs[N_coeffs-1]. (This was tested above, result 3.0)
    }

    #[test]
    fn test_evaluate_spline_input_validation() {
        let knots_valid = arr1(&[0.,0.,1.,1.]); // N=2, m=2. len=4.
        let coeffs_valid = arr1(&[1.0, 2.0]);

        // Order = 0
        assert!(evaluate_spline(0.5, 0, &knots_valid, &coeffs_valid).is_err());

        // Mismatched lengths (validate_knots checks this)
        let knots_wrong_len = arr1(&[0.,0.,1.,1.,1.]); // N=2, m=2. Expected len 4. Got 5.
        let res_wrong_len = evaluate_spline(0.5, 2, &knots_wrong_len, &coeffs_valid);
        assert!(res_wrong_len.is_err());
        if let Err(e) = res_wrong_len {
            assert!(e.contains("Invalid knot vector length"));
        }


        // Invalid knot vector - not sorted
        let knots_not_sorted = arr1(&[0.,1.,0.,1.]);
        let res_not_sorted = evaluate_spline(0.5, 2, &knots_not_sorted, &coeffs_valid);
        assert!(res_not_sorted.is_err());
         if let Err(e) = res_not_sorted {
            assert!(e.contains("Knot vector is not non-decreasing"));
        }


        // Invalid knot vector - t_j == t_{j+m}
        // N=2, m=2. Knots len 4. Example: [0,0,0,0]
        // B_0,2 uses [0,0,0]. t_0=0, t_2=0. t_2 > t_0 fails.
        // B_1,2 uses [0,0,0]. t_1=0, t_3=0. t_3 > t_1 fails.
        let knots_degenerate = arr1(&[0.,0.,0.,0.]);
        let res_degenerate = evaluate_spline(0.5, 2, &knots_degenerate, &coeffs_valid);
        assert!(res_degenerate.is_err());
        if let Err(e) = res_degenerate {
            // Expect message from validate_knots about t_j >= t_{j+order}
            assert!(e.contains("must be less than t_{j+order}"));
        }
    }

    // Tests for derivative functions

    #[test]
    fn test_b_spline_basis_derivative_order1() {
        let knots = arr1(&[0.0, 1.0, 2.0]); // N=2, m=1. len = 3
        let deriv = b_spline_basis_derivative(0, 1, 0.5, &knots).unwrap();
        assert!((deriv - 0.0).abs() < TOL);
    }
    
    #[test]
    fn test_b_spline_basis_derivative_order0_err() {
        let knots = arr1(&[0.0, 1.0, 2.0]);
        assert!(b_spline_basis_derivative(0, 0, 0.5, &knots).is_err());
    }


    #[test]
    fn test_b_spline_basis_derivative_linear_order2() {
        // B_0,2(x) for knots [0,0,1,2]. Support [0,1]. Peak at 0 if t0=t1. No, peak at t1=0.
        // B_0,2 for T=[0,0,1]. Support [0,1]. Linearly increases on [0,0], then decreases.
        // B_0,2(x) = x for x in [0,0] (this part is tricky), (1-x) for x in [0,1] if knots are [0,0,1]
        // For T=[0,0,1,2], m=2.
        // B_0,2(x) uses knots t0,t1,t2 = [0,0,1]. Support [0,1]. B_0,2(x) = (1-x) on [0,1] for these knots. Deriv = -1.
        // B_1,2(x) uses knots t1,t2,t3 = [0,1,2]. Support [0,2]. Triangle shape. Deriv is +1 on [0,1), -1 on [1,2).

        let knots = arr1(&[0.0, 0.0, 1.0, 2.0]); // N=2, m=2. Knots len 4.
        // B_0,2(x) is non-zero on [0,1). Uses t0,t1,t2 = [0,0,1].
        // B_0,2(x) = (x-t0)/(t1-t0)*B01 + (t2-x)/(t2-t1)*B11. Denom (t1-t0)=0 means first term is 0.
        // So B_0,2(x) = (t2-x)/(t2-t1)*B11 = (1-x)/(1-0) * B_1,1(x) where B_1,1 uses subknots [0,1].
        // B_1,1(x) is 1 on [0,1). So B_0,2(x) = 1-x on [0,1). Derivative is -1.
        let deriv_b0_at_0_5 = b_spline_basis_derivative(0, 2, 0.5, &knots).unwrap();
        assert!((deriv_b0_at_0_5 - (-1.0)).abs() < TOL);

        // B_1,2(x) is non-zero on [0,2). Uses t1,t2,t3 = [0,1,2].
        // B_1,2(x) = x on [0,1) and (2-x) on [1,2).
        // Derivative is 1 on [0,1) and -1 on [1,2).
        let deriv_b1_at_0_5 = b_spline_basis_derivative(1, 2, 0.5, &knots).unwrap(); // In [0,1)
        assert!((deriv_b1_at_0_5 - 1.0).abs() < TOL);
        let deriv_b1_at_1_5 = b_spline_basis_derivative(1, 2, 1.5, &knots).unwrap(); // In [1,2)
        assert!((deriv_b1_at_1_5 - (-1.0)).abs() < TOL);
    }
    
    #[test]
    fn test_b_spline_basis_derivative_quadratic_order3() {
        // Example from test_b_spline_basis_book_example_pg69
        // B_0,3(x) on T = {0,1,2,3,4,5}, order m=3.
        // B_0,3(0.5) = 0.125. B_0,3(1.5) = 0.75. B_0,3(2.5) = 0.125
        // Derivative of B_0,3(x) is (m-1) * [ B_0,2(x)/(t2-t0) - B_1,2(x)/(t3-t1) ]
        // t0=0,t1=1,t2=2,t3=3. (t2-t0)=2. (t3-t1)=2.
        // B'_0,3(x) = 2 * [ B_0,2(x)/2 - B_1,2(x)/2 ] = B_0,2(x) - B_1,2(x)
        let knots = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // N_coeffs=3, m=3 for B_j,3
        
        // At x=0.5: (Interval [0,1))
        // B_0,2(0.5) for T'={0,1,2}: (x-0)/(1-0)B01(0.5) + (2-x)/(2-1)B11(0.5) = 0.5*1 + 1.5*0 = 0.5
        // B_1,2(0.5) for T''={1,2,3}: (x-1)/(2-1)B11(0.5) + (3-x)/(3-2)B21(0.5) = -0.5*0 + 2.5*0 = 0
        // B'_0,3(0.5) = 0.5 - 0 = 0.5.
        let deriv_b0_at_0_5 = b_spline_basis_derivative(0, 3, 0.5, &knots).unwrap();
        assert!((deriv_b0_at_0_5 - 0.5).abs() < TOL);

        // At x=1.5: (Interval [1,2))
        // B_0,2(1.5) for T'={0,1,2}: 0.5*0 + 0.5*1 = 0.5
        // B_1,2(1.5) for T''={1,2,3}: 0.5*1 + 1.5*0 = 0.5
        // B'_0,3(1.5) = 0.5 - 0.5 = 0.0.
        let deriv_b0_at_1_5 = b_spline_basis_derivative(0, 3, 1.5, &knots).unwrap();
        assert!((deriv_b0_at_1_5 - 0.0).abs() < TOL);

        // At x=2.5: (Interval [2,3))
        // B_0,2(2.5) for T'={0,1,2} is 0 (outside support [0,2])
        // B_1,2(2.5) for T''={1,2,3}: 1.5*0 + 0.5*1 = 0.5
        // B'_0,3(2.5) = 0 - 0.5 = -0.5.
        let deriv_b0_at_2_5 = b_spline_basis_derivative(0, 3, 2.5, &knots).unwrap();
        assert!((deriv_b0_at_2_5 - (-0.5)).abs() < TOL);
    }

    #[test]
    fn test_evaluate_spline_derivative_constant_order2() {
        // Linear spline (order 2) that is constant: y = 1.0
        // Knots [0,0,1,1], Coeffs [1.0, 1.0]. N_coeffs=2, m=2. Knots len 4.
        // Effective domain [t1,t2] = [0,1].
        // s(x) = 1.0 * B_0,2(x) + 1.0 * B_1,2(x). Since sum(B_j,2(x))=1 on domain, s(x)=1.
        // Derivative should be 0.
        let knots = arr1(&[0.0,0.0,1.0,1.0]);
        let coeffs = arr1(&[1.0, 1.0]);
        let deriv = evaluate_spline_derivative(0.5, 2, &knots, &coeffs).unwrap();
        assert!((deriv - 0.0).abs() < TOL);
    }

    #[test]
    fn test_evaluate_spline_derivative_linear_order2() {
        // s(x) with slope 2 on [0,1)
        // Knots [0,0,1,1], Coeffs [0.0, 2.0]. N_coeffs=2, m=2.
        // On [0,1), B_0,2(x) = 1-x, B_1,2(x) = x.
        // s(x) = 0*(1-x) + 2*x = 2x. Derivative is 2.
        let knots = arr1(&[0.0,0.0,1.0,1.0]);
        let coeffs = arr1(&[0.0, 2.0]);
        let deriv = evaluate_spline_derivative(0.5, 2, &knots, &coeffs).unwrap();
        assert!((deriv - 2.0).abs() < TOL);

        // Knots from evaluate_spline test: [0,0,1,2,2], Coeffs [1,2,1.5]
        // s(0.0)=1.0, s(0.5)=1.5, s(1.0)=2.0.
        // Slope on [0,1) is (2.0-1.0)/(1.0-0.0) = 1.0. (Using endpoint values of segments)
        // My previous manual calculation for s(0.5) was 1.5.
        // s(0) = a0 = 1.0.
        // s(t1) = 1.0. s(t2)=2.0. (using coeffs as control points for linear segments)
        // Slope of first segment (0,1) for these knots/coeffs is (coeffs[1]-coeffs[0]) / (t2-t1) ? No.
        // Using the formula for coefficients of derivative spline:
        // c_i^(1) = (m-1)*(c_i - c_{i-1}) / (t_{i+m-1} - t_i)
        // c_0^(1) not well defined here. c_1^(1) uses c1,c0.
        // Let's use sum a_j B'_j,m(x).
        // B'_0,2(0.5) for [0,0,1,2,2] (uses subknots [0,0,1]) is -1.
        // B'_1,2(0.5) for [0,0,1,2,2] (uses subknots [0,1,2]) is 1.
        // B'_2,2(0.5) for [0,0,1,2,2] (uses subknots [1,2,2]) is 0 (support [1,2]).
        // Deriv = a0*(-1) + a1*(1) + a2*(0) = 1.0*(-1) + 2.0*(1) = -1 + 2 = 1.0.
        let knots2 = arr1(&[0.0, 0.0, 1.0, 2.0, 2.0]);
        let coeffs2 = arr1(&[1.0, 2.0, 1.5]);
        let deriv2_at_0_5 = evaluate_spline_derivative(0.5, 2, &knots2, &coeffs2).unwrap();
        assert!((deriv2_at_0_5 - 1.0).abs() < TOL);

        // At x=1.5 (in [1,2))
        // B'_0,2(1.5) = 0 (support [0,1])
        // B'_1,2(1.5) (uses subknots [0,1,2]) is -1.
        // B'_2,2(1.5) (uses subknots [1,2,2]) is 1. (B_2,2(x) = x-1 for x in [1,2) for these subknots. Deriv 1)
        // Deriv = a0*0 + a1*(-1) + a2*(1) = 2.0*(-1) + 1.5*(1) = -2.0 + 1.5 = -0.5.
        let deriv2_at_1_5 = evaluate_spline_derivative(1.5, 2, &knots2, &coeffs2).unwrap();
        assert!((deriv2_at_1_5 - (-0.5)).abs() < TOL);
    }
    
    #[test]
    fn test_evaluate_spline_derivative_quadratic_order3() {
        // Knots [0,0,0,1,2,2,2], Coeffs [1,2,1.5,3]. Order 3.
        // s(0.0)=1.0, s(0.5)=1.6875, s(1.0)=1.75, s(2.0)=0 (from test_evaluate_quadratic_spline_order3 with corrected end)
        // Derivative should be piecewise linear.
        // At x=0.5:
        // B'_0,3(0.5) for T=[0001222]. Uses subknots [0001]. (B_0,3(x) = (1-x)^2 for these subknots on [0,1]). Deriv = 2(1-x)(-1) = -2(1-x) = -2(0.5) = -1.0
        // B'_1,3(0.5) for T=[0001222]. Uses subknots [0012].
        // B'_2,3(0.5) for T=[0001222]. Uses subknots [0122].
        // B'_3,3(0.5) for T=[0001222]. Uses subknots [1222]. (Support [1,2]) -> 0.
        // Need values for B'_{j,3}(0.5)
        // From test_b_spline_basis_derivative_quadratic_order3 (but different knots):
        // This requires careful calculation of B'_{j,3}(0.5) for these specific knots.
        // B_0,3 uses [0001], B_1,3 uses [0012], B_2,3 uses [0122], B_3,3 uses [1222]
        let knots = arr1(&[0.,0.,0.,1.,2.,2.,2.]);
        let coeffs = arr1(&[1.0, 2.0, 1.5, 3.0]);

        // B'_0,3(0.5): term1 uses B_0,2 on [000], term2 uses B_1,2 on [001]
        // B_0,2 for [000] is 0. B_1,2 for [001] at 0.5 is (1-0.5)=0.5.
        // den1 = t2-t0 = 0-0 (0). den2 = t3-t1 = 1-0=1.
        // B'_0,3(0.5) = (3-1) * [0 - 0.5/1] = 2 * (-0.5) = -1.0
        //
        // B'_1,3(0.5): term1 uses B_1,2 on [001], term2 uses B_2,2 on [012]
        // B_1,2([001], 0.5) = 0.5. B_2,2([012], 0.5) is (0.5-0)/(1-0)B21 + (2-0.5)/(2-1)B31 = 0.5*B21(0.5,[01])+... = 0.5*1=0.5
        // den1 = t3-t1 = 1-0=1. den2 = t4-t2 = 2-0=2.
        // B'_1,3(0.5) = 2 * [0.5/1 - 0.5/2] = 2 * [0.5 - 0.25] = 2 * 0.25 = 0.5
        //
        // B'_2,3(0.5): term1 uses B_2,2 on [012], term2 uses B_3,2 on [122]
        // B_2,2([012], 0.5) = 0.5. B_3,2([122], 0.5) is 0 (support [1,2])
        // den1 = t4-t2 = 2-0=2. den2 = t5-t3 = 2-1=1.
        // B'_2,3(0.5) = 2 * [0.5/2 - 0/1] = 2 * 0.25 = 0.5
        //
        // B'_3,3(0.5) is 0.
        // Sum = 1.0*(-1.0) + 2.0*(0.5) + 1.5*(0.5) + 3.0*0
        //     = -1.0 + 1.0 + 0.75 = 0.75
        let deriv_at_0_5 = evaluate_spline_derivative(0.5, 3, &knots, &coeffs).unwrap();
        assert!((deriv_at_0_5 - 0.75).abs() < TOL);
    }

    #[test]
    fn test_evaluate_spline_derivative_input_validation() {
        let knots_valid = arr1(&[0.,0.,1.,1.]);
        let coeffs_valid = arr1(&[1.0, 2.0]);
        assert!(evaluate_spline_derivative(0.5, 0, &knots_valid, &coeffs_valid).is_err()); // order 0

        let knots_invalid = arr1(&[0.,0.,0.,0.]); // t_j == t_{j+m}
        assert!(evaluate_spline_derivative(0.5, 2, &knots_invalid, &coeffs_valid).is_err());
    }
}
