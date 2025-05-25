use ndarray::Array1;

/// Generates a knot vector with knots placed at quantiles of the data points.
///
/// # Arguments
/// * `data_points` - Sorted covariate data x_i. If not sorted, a sorted copy will be used.
/// * `num_internal_knots` - Number of knots to place between the boundaries.
/// * `order` - Order of the B-spline (m).
///
/// # Returns
/// A `Result` containing the generated knot vector or an error string.
pub fn generate_quantile_knots(
    data_points: &Array1<f64>,
    num_internal_knots: usize,
    order: usize,
) -> Result<Array1<f64>, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }
    if data_points.is_empty() {
        return Err("Data points array cannot be empty.".to_string());
    }

    let mut sorted_data = data_points.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Deduplicate for quantile calculation to reflect unique data points
    let mut unique_sorted_data = sorted_data.clone();
    unique_sorted_data.dedup();

    if num_internal_knots > 0 && unique_sorted_data.len() < num_internal_knots {
         // Warn or error if not enough unique points for requested internal knots.
         // For simplicity, let's allow fewer internal knots if data is not rich enough,
         // by taking all unique points as internal knots if num_internal_knots is too high.
         // However, the problem asks to error if num_internal_knots is too large.
         // "Too large" could mean more than available unique points minus boundary considerations.
         // A strict interpretation: if we cannot pick `num_internal_knots` distinct points
         // strictly between x_min and x_max, it's an error.
         // If unique_sorted_data.len() <= 1 (all points same, or only one point)
         // and num_internal_knots > 0, then we can't pick internal knots.
        if unique_sorted_data.len() <= num_internal_knots && unique_sorted_data.len() <= 1 {
             return Err(format!(
                "Not enough unique data points ({}) to select {} internal knots.",
                unique_sorted_data.len(), num_internal_knots
            ));
        }
    }


    let x_min = *unique_sorted_data.first().unwrap(); // Safe due to earlier empty check
    let x_max = *unique_sorted_data.last().unwrap();   // Safe due to earlier empty check

    if x_min == x_max && num_internal_knots > 0 {
        return Err(format!(
            "Cannot place {} internal knots as all data points are identical ({}).",
            num_internal_knots, x_min
        ));
    }


    let mut knots_vec = Vec::new();

    // Add 'order' repetitions of x_min
    for _ in 0..order {
        knots_vec.push(x_min);
    }

    // Add internal knots
    if num_internal_knots > 0 {
        // Consider only the data points strictly between x_min and x_max for percentile selection
        let internal_candidates: Vec<f64> = unique_sorted_data.iter().copied()
            .filter(|&x| x > x_min && x < x_max)
            .collect();

        if internal_candidates.len() < num_internal_knots {
            return Err(format!(
                "Not enough unique data points strictly between x_min ({}) and x_max ({}) to select {} distinct internal knots. Found {} candidates.",
                x_min, x_max, num_internal_knots, internal_candidates.len()
            ));
        }
        
        // Percentile-based selection
        for i in 1..=num_internal_knots {
            let percentile = i as f64 / (num_internal_knots + 1) as f64;
            let index = (percentile * (internal_candidates.len() -1) as f64).round() as usize;
            // Ensure index is within bounds, though logic should guarantee it.
            let knot_val_index = usize::min(index, internal_candidates.len() - 1);
            knots_vec.push(internal_candidates[knot_val_index]);
        }
        // Sort internal knots just in case rounding leads to out-of-order selection,
        // though with sorted unique_sorted_data and percentile logic, they should be sorted.
        // The section of knots_vec to sort is from index `order` to `order + num_internal_knots - 1`.
        if num_internal_knots > 1 {
             let internal_knots_slice = &mut knots_vec[order..(order + num_internal_knots)];
             internal_knots_slice.sort_by(|a,b| a.partial_cmp(b).unwrap());
        }

    }

    // Add 'order' repetitions of x_max
    for _ in 0..order {
        knots_vec.push(x_max);
    }
    
    // Ensure overall knot vector is non-decreasing (mainly for x_min/x_max against internal knots)
    // This should be guaranteed if internal knots are correctly chosen between x_min and x_max.
    // However, if x_min == x_max, and num_internal_knots = 0, this is fine.
    // If internal knots were somehow outside [x_min, x_max], this would be an issue.
    // The filtering `x > x_min && x < x_max` ensures internal knots are strictly within.

    Ok(Array1::from(knots_vec))
}

/// Generates a knot vector with uniformly spaced knots.
///
/// # Arguments
/// * `x_min` - Minimum value of the range.
/// * `x_max` - Maximum value of the range.
/// * `num_internal_knots` - Number of knots to place between the boundaries.
/// * `order` - Order of the B-spline (m).
///
/// # Returns
/// A `Result` containing the generated knot vector or an error string.
pub fn generate_uniform_knots(
    x_min: f64,
    x_max: f64,
    num_internal_knots: usize,
    order: usize,
) -> Result<Array1<f64>, String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }
    if x_min > x_max { // x_min == x_max is allowed if num_internal_knots = 0
        return Err("x_min cannot be greater than x_max.".to_string());
    }
    if x_min == x_max && num_internal_knots > 0 {
        return Err(format!(
            "Cannot place {} internal knots as x_min ({}) equals x_max ({}).",
            num_internal_knots, x_min, x_max
        ));
    }

    let mut knots_vec = Vec::new();

    // Add 'order' repetitions of x_min
    for _ in 0..order {
        knots_vec.push(x_min);
    }

    // Add internal knots
    if num_internal_knots > 0 {
        let step = (x_max - x_min) / (num_internal_knots + 1) as f64;
        for i in 1..=num_internal_knots {
            knots_vec.push(x_min + i as f64 * step);
        }
    }

    // Add 'order' repetitions of x_max
    for _ in 0..order {
        knots_vec.push(x_max);
    }

    Ok(Array1::from(knots_vec))
}

/// Validates a knot vector.
///
/// # Arguments
/// * `knots` - The knot vector to validate.
/// * `order` - Order of the B-spline (m).
/// * `num_coefficients` - Expected number of basis functions (N).
///
/// # Returns
/// `Ok(())` if the knot vector is valid, or an error string.
pub fn validate_knots(
    knots: &Array1<f64>,
    order: usize,
    num_coefficients: usize,
) -> Result<(), String> {
    if order == 0 {
        return Err("Order (m) must be at least 1.".to_string());
    }
    // Minimum number of knots for order m and N coefficients is N+m.
    // However, N itself depends on knots.len() and order if not given.
    // If N is given, it's a direct check.
    if knots.len() != num_coefficients + order {
        return Err(format!(
            "Invalid knot vector length. Expected {}, got {}. (num_coefficients={}, order={})",
            num_coefficients + order,
            knots.len(),
            num_coefficients,
            order
        ));
    }
    
    // Check for non-decreasing order
    for i in 0..(knots.len() - 1) {
        if knots[i] > knots[i + 1] {
            // Add a small tolerance for floating point comparisons?
            // For now, strict comparison. If issues arise, consider tolerance.
            // if (knots[i] - knots[i+1]) > 1e-9 (or some epsilon)
            return Err(format!(
                "Knot vector is not non-decreasing: t_{}={} > t_{}={}",
                i,
                knots[i],
                i + 1,
                knots[i + 1]
            ));
        }
    }

    // Specific check from definition: t_{j+m} > t_j for B_j,m to be well-defined.
    // This implies that we cannot have m identical knots consecutively
    // if they are involved in the support of a basis function.
    // Example: knots [0,0,0,1,1,1] for m=3.
    // B_0,3 uses t_0 to t_3. t_0=0, t_3=1. t_3 > t_0. (1 > 0) OK.
    // B_1,3 uses t_1 to t_4. t_1=0, t_4=1. t_4 > t_1. (1 > 0) OK.
    // B_2,3 uses t_2 to t_5. t_2=0, t_5=1. t_5 > t_2. (1 > 0) OK.
    // This condition (t_{j+m} > t_j) is required to avoid division by zero in spline basis computation.
    // If knots[j+order-1] - knots[j] == 0 or knots[j+order] - knots[j+1] == 0
    // This is usually stated as t_{i+m-1} != t_i for all i where B_{i,m-1} is part of B_{i,m}
    // More simply, for any basis function B_{j,m}(x) to be non-zero over some interval,
    // we need knots[j+m] > knots[j]. (Support is [t_j, t_{j+m}])
    // This must hold for all j from 0 to num_coefficients-1.
    if num_coefficients > 0 { // Only if there are basis functions to check
        for j in 0..num_coefficients {
            let knot_j = knots[j];
            let knot_j_plus_order = knots[j+order];
            if knot_j_plus_order <= knot_j { // Using <= to catch equality too
                 let idx = j + order;
                 return Err(format!(
                    "Invalid knot configuration: t_j ({}) must be less than t_{{j+order}} ({}). Knot at index {} ({}) <= knot at index {} ({}).",
                    knot_j, knot_j_plus_order, idx, knot_j_plus_order, j, knot_j
                ));
            }
        }
    }


    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    const TOL: f64 = 1e-9;

    // Helper for float array comparison
    fn assert_arr_eq(a: &Array1<f64>, b: &Array1<f64>) {
        assert_eq!(a.len(), b.len(), "Array lengths differ.");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((val_a - val_b).abs() < TOL, "Mismatch at index {}: {} vs {}", i, val_a, val_b);
        }
    }

    #[test]
    fn test_generate_quantile_knots_no_internal() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let knots = generate_quantile_knots(&data, 0, 3).unwrap();
        // Expected: 3x x_min, 3x x_max = [1,1,1, 5,5,5]
        assert_arr_eq(&knots, &arr1(&[1.0, 1.0, 1.0, 5.0, 5.0, 5.0]));
    }

    #[test]
    fn test_generate_quantile_knots_one_internal() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 10.0]); // Median of unique internal [2,3,4] is 3.0
        let knots = generate_quantile_knots(&data, 1, 2).unwrap();
        // data: 1,2,3,4,10. unique_sorted: 1,2,3,4,10. x_min=1, x_max=10.
        // internal_candidates: 2,3,4. num_internal_knots=1.
        // percentile = 1/(1+1) = 0.5. index = (0.5 * (3-1)).round() = 1.
        // internal_candidates[1] = 3.0.
        // Expected: 2x x_min, median, 2x x_max = [1,1, 3, 10,10]
        assert_arr_eq(&knots, &arr1(&[1.0, 1.0, 3.0, 10.0, 10.0]));

        let data_even = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]); // internal [2,3,4,5]
        // internal_candidates: 2,3,4,5. len=4.
        // percentile = 0.5. index = (0.5 * (4-1)).round() = (0.5*3).round() = 1.5.round() = 2.
        // internal_candidates[2] = 4.0
        let knots_even = generate_quantile_knots(&data_even, 1, 2).unwrap();
        assert_arr_eq(&knots_even, &arr1(&[1.0, 1.0, 4.0, 10.0, 10.0]));
    }

    #[test]
    fn test_generate_quantile_knots_two_internal() {
        // data: 1 .. 10. internal candidates: 2,3,4,5,6,7,8,9 (len 8)
        // num_internal_knots = 2.
        // Knot 1: i=1. perc = 1/3. index = (1/3 * 7).round() = 2.33.round() = 2. internal_candidates[2] = 4.0
        // Knot 2: i=2. perc = 2/3. index = (2/3 * 7).round() = 4.66.round() = 5. internal_candidates[5] = 7.0
        let data = Array1::<f64>::range(1.0, 10.1, 1.0); // 1,2,..,10
        let knots = generate_quantile_knots(&data, 2, 3).unwrap();
        // Expected: [1,1,1, 4,7, 10,10,10]
        assert_arr_eq(&knots, &arr1(&[1.0, 1.0, 1.0, 4.0, 7.0, 10.0, 10.0, 10.0]));
    }
    
    #[test]
    fn test_generate_quantile_knots_duplicates_and_sorting() {
        let data = arr1(&[5.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 10.0]);
        // sorted unique: 1,2,3,4,5,10. x_min=1, x_max=10.
        // internal candidates: 2,3,4,5 (len 4)
        // num_internal_knots = 1. index = (0.5 * 3).round() = 2. internal_candidates[2] = 4.0
        let knots = generate_quantile_knots(&data, 1, 2).unwrap();
        assert_arr_eq(&knots, &arr1(&[1.0, 1.0, 4.0, 10.0, 10.0]));
    }

    #[test]
    fn test_generate_quantile_knots_empty_data() {
        let data = arr1(&[]);
        assert!(generate_quantile_knots(&data, 1, 3).is_err());
    }
    
    #[test]
    fn test_generate_quantile_knots_order_zero() {
        let data = arr1(&[1.0, 2.0, 3.0]);
        assert!(generate_quantile_knots(&data, 1, 0).is_err());
    }

    #[test]
    fn test_generate_quantile_knots_too_few_unique_points_for_internal() {
        // Not enough unique points between min and max
        let data = arr1(&[1.0, 1.0, 1.0, 10.0, 10.0]); // x_min=1, x_max=10. No internal candidates.
        assert!(generate_quantile_knots(&data, 1, 2).is_err());

        let data2 = arr1(&[1.0, 2.0, 10.0]); // x_min=1, x_max=10. One internal candidate: 2.0
        assert!(generate_quantile_knots(&data2, 2, 2).is_err()); // Request 2, have 1

        // All points identical
        let data3 = arr1(&[5.0, 5.0, 5.0]);
        assert!(generate_quantile_knots(&data3, 1, 2).is_err());
    }
    
    #[test]
    fn test_generate_quantile_knots_all_points_identical_no_internal() {
        let data = arr1(&[5.0, 5.0, 5.0]);
        let knots = generate_quantile_knots(&data, 0, 3).unwrap();
        // Expected: 3x 5.0, 3x 5.0
        assert_arr_eq(&knots, &arr1(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]));
    }


    #[test]
    fn test_generate_uniform_knots_no_internal() {
        let knots = generate_uniform_knots(1.0, 5.0, 0, 3).unwrap();
        assert_arr_eq(&knots, &arr1(&[1.0, 1.0, 1.0, 5.0, 5.0, 5.0]));
    }

    #[test]
    fn test_generate_uniform_knots_few_internal() {
        let knots = generate_uniform_knots(0.0, 4.0, 3, 2).unwrap();
        // x_min=0, x_max=4. num_internal=3. order=2.
        // step = (4-0)/(3+1) = 1.0
        // internal knots: 0+1*1=1, 0+2*1=2, 0+3*1=3
        // Expected: [0,0, 1,2,3, 4,4]
        assert_arr_eq(&knots, &arr1(&[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0]));
    }
    
    #[test]
    fn test_generate_uniform_knots_xmin_equals_xmax_no_internal() {
        let knots = generate_uniform_knots(5.0, 5.0, 0, 3).unwrap();
        assert_arr_eq(&knots, &arr1(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]));
    }

    #[test]
    fn test_generate_uniform_knots_xmin_equals_xmax_with_internal_err() {
        assert!(generate_uniform_knots(5.0, 5.0, 1, 3).is_err());
    }
    
    #[test]
    fn test_generate_uniform_knots_xmin_greater_than_xmax_err() {
        assert!(generate_uniform_knots(5.0, 1.0, 1, 3).is_err());
    }

    #[test]
    fn test_generate_uniform_knots_order_zero_err() {
        assert!(generate_uniform_knots(1.0, 5.0, 1, 0).is_err());
    }


    #[test]
    fn test_validate_knots_valid() {
        let knots = arr1(&[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]); // N=6, m=3. Knots len = 9.
        // N = len - m = 9 - 3 = 6.
        assert!(validate_knots(&knots, 3, 6).is_ok());
        
        // Example from paper: x_min=0, x_max=1, N_I=3, m=4
        // knots: 0,0,0,0, 0.25,0.5,0.75, 1,1,1,1 (len 11)
        // N = N_I + m = 3+4 = 7. No, N = N_I + 2 (boundary knots) if N_I means internal.
        // Number of basis functions N_coeffs = num_internal_knots + order.
        // For the above: N_coeffs = 3 + 4 = 7.
        // knots.len() = N_coeffs + order = 7 + 4 = 11. This matches.
        let paper_knots = arr1(&[0.0,0.0,0.0,0.0, 0.25,0.5,0.75, 1.0,1.0,1.0,1.0]);
        assert!(validate_knots(&paper_knots, 4, 7).is_ok());
    }

    #[test]
    fn test_validate_knots_invalid_length() {
        let knots = arr1(&[0.0, 0.0, 1.0, 2.0, 2.0]); // N=2, m=3. Expected len 5.
        assert!(validate_knots(&knots, 3, 3).is_err()); // N=3, m=3. Expected len 6.
    }

    #[test]
    fn test_validate_knots_not_sorted() {
        let knots = arr1(&[0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 4.0, 4.0, 4.0]); // N=6, m=3
        assert!(validate_knots(&knots, 3, 6).is_err());
    }
    
    #[test]
    fn test_validate_knots_order_zero() {
        let knots = arr1(&[0.0, 1.0, 2.0]);
        assert!(validate_knots(&knots, 0, 3).is_err());
    }

    #[test]
    fn test_validate_knots_t_j_equals_t_j_plus_m() {
        // N_coeffs = 1, m = 3. knots.len() = 1+3=4
        // knots = [0,0,0,0]. B_0,3 support [t_0, t_3] = [0,0]. This is invalid. t_3 > t_0 required.
        let knots1 = arr1(&[0.0, 0.0, 0.0, 0.0]);
        assert!(validate_knots(&knots1, 3, 1).is_err());

        // N_coeffs = 2, m = 3. knots.len() = 2+3=5
        // knots = [0,0,0,0,1].
        // B_0,3 support [t_0,t_3]=[0,0]. Invalid.
        let knots2 = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(validate_knots(&knots2, 3, 2).is_err());

        // knots = [0,1,1,1,1].
        // B_0,3 support [t_0,t_3]=[0,1]. OK. (knots[3] > knots[0])
        // B_1,3 support [t_1,t_4]=[1,1]. Invalid. (knots[4] <= knots[1])
        let knots3 = arr1(&[0.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(validate_knots(&knots3, 3, 2).is_err());

        // Valid version: N=3, m=3. Knots len = 6.
        // [0,0,0, 1,1,1]. This is Schoenberg's notation for endpoint interpolation.
        // t_0=0, t_1=0, t_2=0, t_3=1, t_4=1, t_5=1
        // N_coeffs = 3 (B0, B1, B2)
        // B_0,3: t_0,t_3. knots[3]>knots[0] (1>0) OK
        // B_1,3: t_1,t_4. knots[4]>knots[1] (1>0) OK
        // B_2,3: t_2,t_5. knots[5]>knots[2] (1>0) OK
        let knots4 = arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert!(validate_knots(&knots4, 3, 3).is_ok());
    }
}
