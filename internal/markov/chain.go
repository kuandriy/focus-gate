package markov

import "sort"

// Transition represents a predicted next topic with its probability.
type Transition struct {
	TopicID     string
	Probability float64
}

// Chain is a sparse Markov transition matrix over topic (tree) IDs.
// Counts[from][to] = number of times the user moved from topic "from" to topic "to".
type Chain struct {
	Counts    map[string]map[string]int `json:"counts"`
	Totals    map[string]int            `json:"totals"` // row sums for O(1) normalization
	LastTopic string                    `json:"lastTopic"`
}

// New creates an empty chain.
func New() *Chain {
	return &Chain{
		Counts: make(map[string]map[string]int),
		Totals: make(map[string]int),
	}
}

// Record increments the transition count from → to.
func (c *Chain) Record(from, to string) {
	if from == "" || to == "" {
		return
	}
	if c.Counts[from] == nil {
		c.Counts[from] = make(map[string]int)
	}
	c.Counts[from][to]++
	c.Totals[from]++
}

// Probability returns P(to | from) = counts[from][to] / totals[from].
// Returns 0 if no data exists.
func (c *Chain) Probability(from, to string) float64 {
	if from == "" || to == "" {
		return 0
	}
	total := c.Totals[from]
	if total == 0 {
		return 0
	}
	return float64(c.Counts[from][to]) / float64(total)
}

// Predict returns the most likely next topic from the given topic.
// Returns "" if no transitions are recorded from this topic.
func (c *Chain) Predict(from string) string {
	row := c.Counts[from]
	if len(row) == 0 {
		return ""
	}
	bestID := ""
	bestCount := 0
	for id, count := range row {
		if count > bestCount {
			bestCount = count
			bestID = id
		}
	}
	return bestID
}

// TopTransitions returns the top N transitions from a topic, sorted by probability descending.
func (c *Chain) TopTransitions(from string, n int) []Transition {
	row := c.Counts[from]
	if len(row) == 0 {
		return nil
	}
	total := c.Totals[from]
	if total == 0 {
		return nil
	}

	ts := make([]Transition, 0, len(row))
	for id, count := range row {
		ts = append(ts, Transition{
			TopicID:     id,
			Probability: float64(count) / float64(total),
		})
	}
	sort.Slice(ts, func(i, j int) bool {
		return ts[i].Probability > ts[j].Probability
	})
	if n > len(ts) {
		n = len(ts)
	}
	return ts[:n]
}

// PruneTopic removes all references to a topic ID (both as source and destination).
func (c *Chain) PruneTopic(topicID string) {
	// Remove outgoing transitions
	if total := c.Totals[topicID]; total > 0 {
		delete(c.Counts, topicID)
		delete(c.Totals, topicID)
	}

	// Remove incoming transitions from all other rows
	for from, row := range c.Counts {
		if count, ok := row[topicID]; ok {
			delete(row, topicID)
			c.Totals[from] -= count
			if c.Totals[from] <= 0 {
				delete(c.Totals, from)
			}
			if len(row) == 0 {
				delete(c.Counts, from)
			}
		}
	}

	// Clear lastTopic if it pointed to the pruned topic
	if c.LastTopic == topicID {
		c.LastTopic = ""
	}
}

// TransitionCount returns the total number of recorded transitions.
func (c *Chain) TransitionCount() int {
	total := 0
	for _, t := range c.Totals {
		total += t
	}
	return total
}
