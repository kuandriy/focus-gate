package text

// LevenshteinCutoff returns the edit distance between a and b, bailing out
// early if the distance is known to exceed cutoff.
//
// The return value is the true distance when it is ≤ cutoff, and a value
// strictly greater than cutoff otherwise. Callers should treat the result
// as a threshold test, not a precise distance when > cutoff.
//
// The algorithm uses the standard two-row dynamic-programming formulation,
// operating on runes (not bytes) so multi-byte unicode sequences count as
// one edit unit. Length-difference prefilter:
//
//	if |len(a) - len(b)| > cutoff, return cutoff + 1 immediately.
//
// Each row is additionally pruned: once every cell in a row exceeds cutoff,
// no subsequent row can re-enter the ≤ cutoff range, so we return early.
// This keeps the inner loop cheap on unrelated word pairs — exactly the
// common case for per-prompt canonicalization.
func LevenshteinCutoff(a, b string, cutoff int) int {
	ra := []rune(a)
	rb := []rune(b)
	la := len(ra)
	lb := len(rb)

	// Trivial cases: either string empty.
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}

	// Length-difference prefilter. If the two strings differ in length by
	// more than cutoff, they are guaranteed to require more than cutoff edits.
	if diff := la - lb; diff > cutoff || -diff > cutoff {
		return cutoff + 1
	}

	// Ensure ra is the shorter to keep the working rows small.
	if la > lb {
		ra, rb = rb, ra
		la, lb = lb, la
	}

	prev := make([]int, la+1)
	curr := make([]int, la+1)
	for i := 0; i <= la; i++ {
		prev[i] = i
	}

	for j := 1; j <= lb; j++ {
		curr[0] = j
		minInRow := curr[0]
		for i := 1; i <= la; i++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			// Deletion, insertion, substitution.
			del := prev[i] + 1
			ins := curr[i-1] + 1
			sub := prev[i-1] + cost
			v := del
			if ins < v {
				v = ins
			}
			if sub < v {
				v = sub
			}
			curr[i] = v
			if v < minInRow {
				minInRow = v
			}
		}
		// Row-level early exit: if even the minimum cell in this row already
		// exceeds cutoff, no cell in any subsequent row can fall back to ≤
		// cutoff (distances are monotonically related to row index from here).
		if minInRow > cutoff {
			return cutoff + 1
		}
		prev, curr = curr, prev
	}

	return prev[la]
}
